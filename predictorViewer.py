# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/19 16:00
desc: 定位结果的显示工具
'''
import copy
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from filterpy.stats import plot_covariance
import OpenGL.GL as ogl
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QWidget
from pyqtgraph.dockarea import DockArea, Dock


def findPeakValley(data, noiseStd):
    '''
    寻峰算法:
    1、对连续三个点a, b, c，若a<b && b>c，且b>E0，则b为峰点；若a>b && b<c，且b<E0，则b为谷点
    2、保存邻近的峰点和谷点，取x=±n个点内的最大或最小值作为该区段的峰点或谷点
    :param data: 【pd】读取的原始数据
    :param noiseStd: 【float】噪声值
    :param Vpp: 【float】原始数据的平均峰峰值
    :return:
    '''
    dataSize = len(data)
    E0 = np.array(data).mean()
    #print("dataSize={}, E0={:.2f}".format(dataSize, E0))
    # startIndex = data._stat_axis._start
    # 找出满足条件1的峰和谷
    peaks, valleys = [], []
    for i in range(1, dataSize - 1):
        d1, d2, d3 = data[i - 1], data[i], data[i + 1]  # 用于实时获取的数据
        point = (i + 1, d2)
        if d1 < d2 and d2 >= d3 and d2 > E0 + 3 * noiseStd:
            if not peaks or i - peaks[-1][0] > 9:  # 第一次遇到峰值或距离上一个峰值超过9个数
                peaks.append(point)
            elif peaks[-1][1] < d2:  # 局部区域有更大的峰值
                peaks[-1] = point
        elif d1 > d2 and d2 <= d3 and d2 < E0 - 3 * noiseStd:
            if not valleys or i - valleys[-1][0] > 9:  # 第一次遇到谷值或距离上一个谷值超过9个数
                valleys.append(point)
            elif valleys[-1][1] > d2:  # 局部区域有更小的谷值
                valleys[-1] = point

    peaks_y = [peak[1] for peak in peaks]
    valleys_y = [valley[1] for valley in valleys]

    peakMean = sum(peaks_y) / len(peaks_y) if len(peaks_y) else 0
    valleyMean = sum(valleys_y) / len(valleys_y) if len(valleys_y) else 0
    return peakMean - valleyMean

def q2R(q):
    '''
    从四元数求旋转矩阵
    :param q: 四元数
    :return: R 旋转矩阵
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3,     2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3,     1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2,     2 * q2 * q3 + 2 * q0 * q1,     1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])
    return R

def q2ua(q):
    '''
    从四元数求旋转向量和旋转角
    :param q:
    :return:
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    angle = 2 * math.acos(q0)
    u = np.array([q1, q2, q3]) / math.sin(0.5 * angle) if angle else np.array([0, 0, 1])
    return u, angle * 57.3

def q2Euler(q):
    '''
    从四元数求欧拉角
    :param q: 四元数
    :return: 【np.array】 [pitch, roll, yaw]
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    roll = math.atan2(2 * q0 * q1 + 2 * q2 * q3, 1 - 2 * q1 * q1 - 2 * q2 * q2)
    pitch = math.asin(2 * q0 * q2 - 2 * q3 * q1)
    yaw = math.atan2(2 * q0 * q3 + 2 * q1 * q2, 1 - 2 * q2 * q2 - 2 * q3 * q3)
    return np.array([pitch, roll, yaw]) * 57.3

def parseState(state):
    '''
    从位姿状态中获取旋转矢量，并提取位置和欧拉角
    :param state: 【np.array】/【se3】位姿
    :return: 【list】[pos, angle, uAxis, euler]
    '''
    if isinstance(state, np.ndarray):
        pos, q = np.array(state[:3]) * 0.1, state[3:7]
        uAxis, angle = q2ua(q)
        euler = q2Euler(q)
    else:
        pos = state.exp().matrix()[:3, 3] * 0.1
        angle = np.linalg.norm(state.w[:3])
        uAxis = state.w[:3] / angle if angle else np.array([0, 0, 1])
        angle *= 57.3
        euler = q2Euler(state.quaternion())
    return [pos, angle, uAxis, euler]


def plotLM(residual_memory, us):
    '''
    描绘LM算法的残差和u值（LM算法的参数）曲线
    :param residual_memory: 【list】残差列表
    :param us: 【list】u值列表
    :return:
    '''
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # plt.plot(residual_memory)
    for ax in [ax1, ax2]:
        ax.set_xlabel("iter")
    ax1.set_ylabel("residual")
    ax1.semilogy(residual_memory)
    ax2.set_xlabel("iter")
    ax2.set_ylabel("u")
    ax2.semilogy(us)
    plt.show()


# plt.axis('auto')   # 坐标轴自动缩放

def plotP(predictor, state, index, plotType):
    '''
    描绘UKF算法中误差协方差yz分量的变化过程
    :param state0: 【np.array】预测状态 （7，）
    :param state: 【np.array】真实状态 （7，）
    :param index: 【int】算法的迭代步数
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :return:
    '''
    x, y = plotType
    state_copy = state.copy()  # 浅拷贝真实值，因为后面会修改state
    xtruth = state_copy[:3]  # 获取坐标真实值
    mtruth = q2R(state_copy[3: 7])[:, -1]  # 获取姿态真实值，并转换为z方向的矢量

    pos, q = predictor.ukf.x[:3].copy(), predictor.ukf.x[3: 7]  # 获取预测值，浅拷贝坐标值
    em = q2R(q)[:, -1]
    if plotType == (0, 1):
        plt.ylim(-0.2, 0.4)
        plt.axis('equal')  # 坐标轴按照等比例绘图
    elif plotType == (1, 2):
        xtruth[1] += index * 0.1
        pos[1] += index * 0.1
    else:
        raise Exception("invalid plotType")

    P = predictor.ukf.P[x: y+1, x: y+1]  # 坐标的误差协方差
    plot_covariance(mean=pos[x: y+1], cov=P, fc='g', alpha=0.3, title='胶囊定位过程仿真')
    plt.text(pos[x], pos[y], int(index), fontsize=9)
    plt.plot(xtruth[x], xtruth[y], 'ro')  # 画出真实值
    plt.text(xtruth[x], xtruth[y], int(index), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos[x] + em[x] * scale, pos[y] + em[y] * scale), xytext=(pos[x], pos[y]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[x] + mtruth[x] * scale, xtruth[y] + mtruth[y] * scale),
                 xytext=(xtruth[x], xtruth[y]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))

    # 添加坐标轴标识
    plt.xlabel('{}/m'.format('xyz'[x]))
    plt.ylabel('{}/m'.format('xyz'[y]))
    # 添加网格线
    plt.gca().grid(b=True)
    # 增加固定时间间隔
    plt.pause(0.05)

def plotPos(state0, state, index, plotType):
    '''
    描绘预测位置的变化过程
    :param state0: 【np.array】预测状态 （7，）
    :param state: 【np.array】真实状态 （7，）
    :param index: 【int】算法的迭代步数
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :return:
    '''
    x, y = plotType
    state_copy = state.copy()  # 浅拷贝真实值，因为后面会修改state
    xtruth = state_copy[:3]  # 获取坐标真实值
    mtruth = q2R(state_copy[3: 7])[:, -1]  # 获取姿态真实值，并转换为z方向的矢量

    pos, q = state0[:3].copy(), state0[3:]    # 获取预测值，浅拷贝坐标值
    em = q2R(q)[:, -1]
    if plotType == (0, 1):
        # 添加坐标轴标识
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.axis('equal')    # 坐标轴按照等比例绘图
        plt.ylim(-0.2, 0.5)
        # plt.gca().set_aspect('equal', adjustable='box')
    elif plotType == (1, 2):
        xtruth[1] += index
        pos[1] += index
        # 添加坐标轴标识
        plt.xlabel('y/m')
        plt.ylabel('z/m')
    else:
        raise Exception("invalid plotType")

    plt.plot(pos[x], pos[y], 'b+')  # 仅描点
    plt.text(pos[x], pos[y], int(index), fontsize=9)
    plt.plot(xtruth[x], xtruth[y], 'ro')  # 画出真实值
    plt.text(xtruth[x], xtruth[y], int(index), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos[x] + em[x] * scale, pos[y] + em[y] * scale), xytext=(pos[x], pos[y]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[x] + mtruth[x] * scale, xtruth[y] + mtruth[y] * scale),
                 xytext=(xtruth[x], xtruth[y]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))

    plt.gca().grid(b=True)
    plt.pause(0.05)

def plotErr(x, y, z, contourBar, titleName):
    '''
    描绘误差分布的等高线图
    :param x: 【np.array】误差分布的x变量 (n, n)
    :param y: 【np.array】误差分布的y变量 (n, n)
    :param z: 【np.array】误差分布的结果 (n, n)
    :param contourBar: 【np.array】等高线的刻度条
    :param titleName: 【string】图的标题名称
    :return:
    '''
    plt.title(titleName)
    plt.xlabel('x/mm')
    plt.ylabel('y/mm')
    plt.tick_params(labelsize=10)
    plt_contourf = plt.contourf(x, y, z, contourBar, cmap='jet', extend='both')    # 填充等高线内区域
    cmap = copy.copy(plt_contourf.get_cmap())
    cmap.set_over('red')     # 超过contourBar的上限就填充为red
    cmap.set_under('blue')     # 低于contourBar的下限就填充为blue
    plt_contourf.changed()

    cntr = plt.contour(x, y, z, contourBar, colors='black', linewidths=0.5)    # 描绘等高线轮廓
    plt.clabel(cntr, inline_spacing=1, fmt='%.2f', fontsize=8, colors='black')     # 标识等高线的数值
    plt.show()

def plotTrajectory(stateLine, stateMP, sensor_std):
    '''
    描绘轨迹预估图
    :param stateLine: 【np.array】真实状态的轨迹
    :param stateMP: 【np.array】预估的状态
    :param sensor_std: 【float】传感器噪声，此处指感应电压的采样噪声[μV]
    :return:
    '''
    plt.title('sensor_std={}'.format(sensor_std))
    plt.axis('equal')  # 坐标轴按照等比例绘图
    plt.gca().grid(b=True)
    plt.xlabel('x/mm')
    plt.ylabel('y/mm')
    plt.plot(stateLine[:, 0], stateLine[:, 1],'r')
    plt.plot(stateMP[:, 0], stateMP[:, 1], 'b--')
    plt.show()

class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, X, Y, Z, text):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(int(self.X), int(self.Y), int(self.Z), self.text)


class Custom3DAxis(gl.GLAxisItem):
    """Class defined to extend 'gl.GLAxisItem'."""

    def __init__(self, parent, color=(1, 2, 3, 4)):
        gl.GLAxisItem.__init__(self)
        self.parent = parent
        self.c = color
        self.ticks = list(range(-20, 25, 10))
        self.setSize(x=40, y=40, z=20)
        self.add_labels()
        self.add_tick_values(xticks=self.ticks, yticks=self.ticks, zticks=[0, 5, 10, 15, 20])
        self.addArrow()

    def add_labels(self):
        """Adds axes labels."""
        x, y, z = self.size()
        x *= 0.5
        y *= 0.5
        # X label
        self.xLabel = CustomTextItem(X=x + 0.5, Y=-y / 10, Z=-z / 10, text="X(cm)")
        self.xLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.xLabel)
        # Y label
        self.yLabel = CustomTextItem(X=-x / 10, Y=y + 0.5, Z=-z / 10, text="Y(cm)")
        self.yLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.yLabel)
        # Z label
        self.zLabel = CustomTextItem(X=-x / 10, Y=-y / 10, Z=z + 1, text="Z(cm)")
        self.zLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.zLabel)

    def add_tick_values(self, xticks=None, yticks=None, zticks=None):
        """Adds ticks values."""
        x, y, z = self.size()
        xtpos = np.linspace(-0.5 * x, 0.5 * x, len(xticks))
        ytpos = np.linspace(-0.5 * y, 0.5 * y, len(yticks))
        ztpos = np.linspace(0, z, len(zticks))
        # X label
        for i, xt in enumerate(xticks):
            val = CustomTextItem(X=xtpos[i], Y=1, Z=0, text=str(xt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)
        # Y label
        for i, yt in enumerate(yticks):
            val = CustomTextItem(X=1, Y=ytpos[i], Z=0, text=str(yt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)
        # Z label
        for i, zt in enumerate(zticks):
            val = CustomTextItem(X=0, Y=1, Z=ztpos[i], text=str(zt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)

    def addArrow(self):
        arrowXYZ = 20
        # add X axis arrow
        arrowXData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowX = gl.GLMeshItem(meshdata=arrowXData, color=(0, 0, 1, 0.6), shader='balloon', glOptions='opaque')
        arrowX.rotate(90, 0, 1, 0)
        arrowX.translate(arrowXYZ, 0, 0)
        self.parent.addItem(arrowX)
        # add Y axis arrow
        arrowYData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowY = gl.GLMeshItem(meshdata=arrowXData, color=(1, 0, 1, 0.6), shader='balloon', glOptions='opaque')
        arrowY.rotate(270, 1, 0, 0)
        arrowY.translate(0, arrowXYZ, 0)
        self.parent.addItem(arrowY)
        # add Z axis arrow
        arrowZData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowZ = gl.GLMeshItem(meshdata=arrowXData, color=(0, 1, 0, 0.6), shader='balloon', glOptions='opaque')
        arrowZ.translate(0, 0, arrowXYZ)
        self.parent.addItem(arrowZ)

    def paint(self):
        self.setupGLState()
        if self.antialias:
            ogl.glEnable(ogl.GL_LINE_SMOOTH)
            ogl.glHint(ogl.GL_LINE_SMOOTH_HINT, ogl.GL_NICEST)
        ogl.glBegin(ogl.GL_LINES)

        x, y, z = self.size()
        # Draw Z
        ogl.glColor4f(0, 1, 0, 10.6)  # z is green
        ogl.glVertex3f(0, 0, 0)
        ogl.glVertex3f(0, 0, z)
        # Draw Y
        ogl.glColor4f(1, 0, 1, 10.6)  # y is grape
        ogl.glVertex3f(0, -0.5 * y, 0)
        ogl.glVertex3f(0, 0.5 * y, 0)
        # Draw X
        ogl.glColor4f(0, 0, 1, 10.6)  # x is blue
        ogl.glVertex3f(-0.5 * x, 0, 0)
        ogl.glVertex3f(0.5 * x, 0, 0)
        ogl.glEnd()


def track3D(state, qList=None, tracker=None):
    '''
    描绘目标状态的3d轨迹
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
    pos0 = np.array([[0, 0, 0]]) * 0.1

    pos, angle, uAxis, euler = parseState(state)
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
    # 记录位置和姿态的历史值
    pts = pos.reshape(1, 3)
    eulers = euler.reshape(1, 3)

    z = []
    accData = []

    def update():
        nonlocal i, pts, state, eulers, accData

        if qList:   # 从队列中获取实测数据，并更新姿态
            qADC, qGyro, qAcc = qList
            if not qGyro.empty():
                qGyro.get()
            if not qAcc.empty():
                accData = qAcc.get()

            if not qADC.empty():
                adcV = qADC.get()
                vm = findPeakValley(adcV, 4e-6) * 0.5
                if vm:
                    z.append(vm * 1e6)
            if len(z) == 16:
                if accData:
                    for i in range(3):
                        z.append(accData[i])
                    tracker.LM(z)
                    z.clear()

        # update position and orientation
        pos, angle, uAxis, euler = parseState(tracker.state)
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
        timeText.setText(" total time = {:.3f}s\n compute time = {:.3f}s".format(tracker.totalTime, tracker.compTime))
        iterText.setText(" iter times = " + str(tracker.iter))
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(20)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    state = np.array([0, 0, 215, 1, 0, 0, 0, 0.1, 2])
    track3D(state)