# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/12/16 14:31
desc: 基于李代数LM非线性优化，实现内置式磁定位
'''
import time
from queue import Queue
import sys
from multiprocessing.dummy import Process

import numpy as np
import OpenGL.GL as ogl
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QWidget
from pyqtgraph.dockarea import DockArea, Dock

from Lie import se3
from coilArray import CoilArray
from predictorViewer import Custom3DAxis, q2Euler
from readData import readRecData, findPeakValley


class Predictor:
    printBool = False

    def __init__(self, state, currents):
        '''
        初始化定位类
        :param state: 【se3】初始位姿
        :param currents: 【list】 发射端电流幅值[A]
        '''
        self.state = state
        self.n = 6
        self.currents = currents
        self.coils = CoilArray(np.array(currents))
        self.m = self.coils.coilNum + 3
        self.t0 = time.time()
        self.totalTime = 0
        self.compTime = 0
        self.iter = 0

    def h(self, state):
        '''
        观测函数
        :return: 【np.array】观测值
        '''
        pos = state.exp().matrix()[:3, 3]
        R = state.exp().matrix()[:3, :3]
        dArray0 = pos - self.coils.coilArray
        em2 = R[2, :3]

        E = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            E[i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1, em2=em2, ii=self.currents[i])
        return E

    def hh(self, state):
        """
        线圈+IMU的观测方程
        :param state: 预估的状态量 (n, )
        :return: E+A 感应电压 [1e-6V] + 方向矢量[1] (m, )
        """
        pos = state.exp().matrix()[:3, 3]
        R = state.exp().matrix()[:3, :3]
        dArray0 = pos - self.coils.coilArray
        em2 = R[:3, 2]

        EA = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            EA[i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1, em2=em2, ii=self.currents[i])

        EA[-3] = -1000 * em2[-3]  # x反向
        EA[-2] = -1000 * em2[-2]  # y反向
        EA[-1] = 1000 * em2[-1]   # z正向
        return EA

    def residual(self, state, measureData):
        '''
        计算残差
        :param measureData:  【np.array】观测值
        :return: 【np.array】观测值 - 预测值
        '''
        eastData = self.hh(state)
        return measureData - eastData

    def derive(self, param_index):
        """
        指定状态量的偏导数
        :param param_index: 第几个状态量
        :return: 偏导数 (m, )
        """
        state1 = self.state.vector().copy()
        state2 = self.state.vector().copy()
        delta = 0.0001
        state1[param_index] += delta
        state2[param_index] -= delta
        data_est_output1 = self.hh(se3(vector=state1))
        data_est_output2 = self.hh(se3(vector=state2))
        return 0.5 * (data_est_output1 - data_est_output2) / delta

    def jacobian(self):
        '''
        计算预估状态的雅可比矩阵
        :return: 【np.array (m, n)】雅可比矩阵
        '''
        J = np.zeros((self.m, self.n))
        for pi in range(self.n):
            J[:, pi] = self.derive(pi)
        return J

    def get_init_u(self, A, tao):
        """
        确定u的初始值
        :param A: 【np.array】 J.T * J (n, n)
        :param tao: 【float】常量
        :return: 【int】u
        """
        Aii = []
        for i in range(0, self.n):
            Aii.append(A[i, i])
        u = tao * max(Aii)
        return u

    def LM(self, measureData, maxIter=50):
        """
        Levenberg–Marquardt优化算法的主体
        :param measureData: 【np.array】观测值 (m, )
        :param maxIter: 最大迭代次数
        :return: 【np.array】优化后的状态 (7, )
        """
        tao = 1e-3
        eps_stop = 1e-9
        eps_step = 1e-4
        eps_residual = 1e-3

        t0 = time.time()
        res = self.residual(self.state, measureData)
        J = self.jacobian()
        A = J.T.dot(J)
        g = J.T.dot(res)
        u = self.get_init_u(A, tao)  # set the init u
        # u = 100
        v = 2
        rou = 0
        mse = 0

        for i in range(maxIter):
            i += 1
            while True:

                Hessian_LM = A + u * np.eye(self.n)  # calculating Hessian matrix in LM
                step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
                if np.linalg.norm(step) <= eps_step:
                    self.stateOut(t0, i, mse, 'threshold_step')
                    return self.state
                newState = se3(vector=self.state.vector() + step)   # 先将se3转换成数组相加，再变回李代数，这样才符合LM算法流程
                newRes = self.residual(newState, measureData)
                mse = np.linalg.norm(res) ** 2
                newMse = np.linalg.norm(newRes) ** 2
                rou = (mse - newMse) / (step.T.dot(u * step + g))
                if rou > 0:
                    self.state = newState
                    res = newRes
                    J = self.jacobian()
                    A = J.T.dot(J)
                    g = J.T.dot(res)
                    u *= max(1 / 3, 1 - (2 * rou - 1) ** 3)
                    v = 2

                    stop = (np.linalg.norm(g, ord=np.inf) <= eps_stop) or (mse <= eps_residual)
                    if stop:
                        if np.linalg.norm(g, ord=np.inf) <= eps_stop:
                            self.stateOut(t0, i, mse, 'threshold_stop')
                        if mse <= eps_residual:
                            self.stateOut(t0, i, mse, 'threshold_residual')
                        return self.state
                    else:
                        break
                else:
                    u *= v
                    v *= 2
            if i == maxIter:
                self.stateOut(t0, i, mse, 'maxIter_step')

    def stateOut(self, t0, i, mse, printStr):
        '''
        输出算法的中间结果
        :param t0: 【float】 时间戳
        :param i: 【int】迭代步数
        :param mse: 【float】残差
        :return:
        '''
        self.compTime = time.time() - t0
        self.totalTime = time.time() - self.t0
        self.t0 = time.time()
        self.iter = i

        if not self.printBool:
            return
        
        pos = self.state.exp().matrix()[:3, 3]
        em = self.state.exp().matrix()[:3, 2]
        pos = np.round(pos, 1)
        em = np.round(em, 3)
        print(printStr)
        print('i={}, pos={}mm, se3={}, timeConsume={:.3f}s, cost={:.3e}'.format(i, pos, np.round(self.state.w, 3), self.compTime, mse))

    def track3D(self):
        '''
        运行真实的定位程序，描绘目标状态的3d轨迹
        :param state: 【np.array】目标的状态
        :return:
        '''
        app = QtGui.QApplication([])
        qWidget = QWidget()
        qWidget.setWindowTitle("磁定位显示界面")
        qWidget.resize(800, 600)
        qWidget.GLDock = Dock("\n3D位姿\n", size=(600, 500))

        w = gl.GLViewWidget()
        qWidget.GLDock.addWidget(w)

        area = DockArea()
        area.addDock(qWidget.GLDock, 'left')

        # w.setWindowTitle('3d trajectory')
        # w.resize(600, 500)
        # instance of Custom3DAxis
        axis = Custom3DAxis(w, color=(0.6, 0.6, 0.2, .6))
        w.addItem(axis)
        w.opts['distance'] = 75
        w.opts['center'] = pg.Vector(0, 0, 0)
        # add xy grid
        gx = gl.GLGridItem()
        gx.setSize(x=40, y=40, z=10)
        gx.setSpacing(x=5, y=5)
        w.addItem(gx)

        h = QtGui.QHBoxLayout()
        qWidget.setLayout(h)
        h.addWidget(area)

        # trajectory line
        pos0 = np.array([[0, 0, 200]]) * 0.1
        pos = self.state.exp().matrix()[:3, 3] * 0.1
        angle = np.linalg.norm(self.state.w[:3])
        uAxis = self.state.w[:3] / angle if angle else np.array([0, 0, 1])
        angle *= 57.3
        euler = q2Euler(self.state.quaternion())
        track0 = np.concatenate((pos0, pos.reshape(1, 3)))
        plt = gl.GLLinePlotItem(pos=track0, width=2, color=(1, 0, 0, 0.6))
        w.addItem(plt)
        # orientation arrow
        sphereData = gl.MeshData.sphere(rows=20, cols=20, radius=0.6)
        sphereMesh = gl.GLMeshItem(meshdata=sphereData, smooth=True, shader='shaded', glOptions='opaque')
        w.addItem(sphereMesh)
        ArrowData = gl.MeshData.cylinder(rows=20, cols=20, radius=[0.5, 0], length=1.5)
        ArrowMesh = gl.GLMeshItem(meshdata=ArrowData, smooth=True, color=(1, 0, 0, 0.6), shader='balloon',
                                glOptions='opaque')
        ArrowMesh.rotate(angle, uAxis[0], uAxis[1], uAxis[2])
        w.addItem(ArrowMesh)
        # w.setWindowTitle('position={}cm, pitch={:.0f}\u00b0, roll={:.0f}\u00b0, yaw={:.0f}\u00b0,'
        # .format(np.round(pos, 1), euler[0], euler[1], euler[2]))
        #w.show()

        # add Position
        posDock = Dock("\n坐标/cm\n", size=(160, 10))
        posText = QtGui.QLabel()
        posText.setText(" x = {} \u00b1 {}\n\n y = {} \u00b1 {}\n z = {} \u00b1 {}".format(pos0[0, 0], 0, pos0[0, 1], 0, pos0[0, 2], 0))
        posDock.addWidget(posText)
        area.addDock(posDock, 'right')

        # add euler angles
        eulerDock = Dock("\n姿态角/\u00b0\n", size=(160, 10))
        eulerText = QtGui.QLabel()
        eulerText.setText("pitch = {:.0f} \u00b1 {}\n\n roll = {:.0f} \u00b1 {}\n yaw = {:.0f} \u00b1 {}".format(euler[0], 0, euler[1], 0, euler[2], 0))
        eulerDock.addWidget(eulerText)
        area.addDock(eulerDock, 'bottom', posDock)

        # add time cost
        timeDock = Dock("\n耗时/s\n", size=(160, 10))
        timeText = QtGui.QLabel()
        timeText.setText("total time: 0\ncompute time: 0")
        timeDock.addWidget(timeText)
        area.addDock(timeDock, 'bottom', eulerDock)

        # add iter times
        iterDock = Dock("\n迭代次数\n", size=(160, 10))
        iterText = QtGui.QLabel()
        iterText.setText("iter: 0")
        iterDock.addWidget(iterText)
        area.addDock(iterDock, 'bottom', timeDock)

        qWidget.show()

        i = 1
        pts = pos.reshape(1, 3)
        eulers = euler.reshape(1, 3)

        
        # 读取接收端数据
        qADC, qGyro, qAcc = Queue(), Queue(), Queue()
        z = []
        accData = []
        procReadRec = Process(target=readRecData, args=(qADC, qGyro, qAcc))
        procReadRec.daemon = True
        procReadRec.start()
        time.sleep(0.5)

        def update():
            nonlocal i, pts, eulers, accData
            # run LM predictor
            if not qGyro.empty():
                qGyro.get()
            if not qAcc.empty():
                accData = qAcc.get()
        
            if not qADC.empty():
                adcV = qADC.get()
                vm = findPeakValley(adcV, 0, 4e-6) * 0.5
                if vm:
                    z.append(vm * 1e6)
            if len(z) == 16:
                if accData:
                    for i in range(3):
                        z.append(accData[i])
                    self.LM(z)
                    z.clear()

            # update position and orientation
            pos = self.state.exp().matrix()[:3, 3] * 0.1
            angle = np.linalg.norm(self.state.w[:3])
            uAxis = self.state.w[:3] / angle if angle else np.array([0, 0, 1])
            angle *= 57.3
            euler = q2Euler(self.state.quaternion())
            pt = pos.reshape(1, 3)
            et = euler.reshape(1, 3)
            if pts.size < 150:
                pts = np.concatenate((pts, pt))
                eulers = np.concatenate((eulers, et))
            else:
                pts = np.concatenate((pts[-50:, :], pt))
                eulers = np.concatenate((eulers[-50:, :], et))
            plt.setData(pos=pts)

            stdPosX = pts[:, 0].std()
            stdPosY = pts[:, 1].std()
            stdPosz = pts[:, 2].std()
            stdPitch = eulers[:, 0].std()
            stdRoll = eulers[:, 1].std()
            stdYaw = eulers[:, 2].std()

            # update gui
            ArrowMesh.resetTransform()
            sphereMesh.resetTransform()
            ArrowMesh.rotate(angle, uAxis[0], uAxis[1], uAxis[2])
            ArrowMesh.translate(*pos)
            sphereMesh.translate(*pos)
            
            # update state
            posText.setText(" x = {:.2f} \u00b1 {:.2f}\n\n y = {:.2f} \u00b1 {:.2f}\n\n z = {:.2f} \u00b1 {:.2f}".format(pos[0], stdPosX, pos[1], stdPosY, pos[2], stdPosz))
            eulerText.setText(" pitch = {:.0f} \u00b1 {:.0f}\n\n roll = {:.0f} \u00b1 {:.0f}\n\n yaw = {:.0f} \u00b1 {:.0f}".format(euler[0], stdPitch, euler[1], stdRoll, euler[2], stdYaw))
            timeText.setText(" total time = {:.3f}s\n compute time = {:.3f}s".format(self.totalTime, self.compTime))
            iterText.setText(" iter times = " + str(self.iter))
            i += 1

        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(50)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()



if __name__ == '__main__':
    state = se3(vector=np.array([0, 0, 0, 0, 0, 200]))
    currents = [2.15, 2.18, 2.25, 2.36, 2.28, 2.25, 2.25, 2.33, 2.22, 2.35, 2.32, 2.3, 2.3, 2.38, 2.39, 2.27]
    pred = Predictor(state, currents)
    pred.track3D()
