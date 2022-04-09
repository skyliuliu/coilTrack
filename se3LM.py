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

import numpy as np

from Lie import se3
from coilArray import CoilArray
from predictorViewer import track3D


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
                return self.state

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


if __name__ == '__main__':
    state = se3(vector=np.array([0, 0, 0, 0, 0, 200]))
    currents = [2.15, 2.18, 2.25, 2.36, 2.28, 2.25, 2.25, 2.33, 2.22, 2.35, 2.32, 2.3, 2.3, 2.38, 2.39, 2.27]
    pred = Predictor(state, currents)
    track3D(state, qList=[Queue(), Queue(), Queue()], tracker=pred)
