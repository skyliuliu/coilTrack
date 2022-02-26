# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/12/16 14:31
desc: 基于李代数LM非线性优化，实现内置式磁定位
'''
import datetime
import math

from Lie import *
from coilArray import CoilArray


class Predictor:

    def __init__(self, state, stateX, currents):
        '''
        初始化定位类
        :param state: 【se3】初始位姿
        :param stateX: 【se3】真实位姿
        '''
        self.state = state
        self.stateX = stateX
        self.n = 6
        self.currents = currents
        self.coils = CoilArray(np.array(currents))
        self.m = self.coils.coilNum

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
            E[i] = self.coils.inducedVolatage(d=d, em2=em2, ii=self.currents[i])
        return E

    def generateData(self, std):
        '''
        生成模拟数据
        :param Std: 传感器输出值的标准差
        :return:
        '''
        midData = self.h(self.stateX)
        pos = self.stateX.exp().matrix()[:3, 3]
        ez = self.stateX.exp().matrix()[2, :3]
        print('turth: pos={}, ez={}'.format(np.round(pos, 3), np.round(ez, 3)))
        simData = np.zeros(self.m)
        for j in range(self.m):
            simData[j] = np.random.normal(midData[j], std, 1)
        self.measureData = simData
        print('sensor_mid={}'.format(np.round(midData[:], 3)))
        print('sensor_sim={}'.format(np.round(simData[:], 3)))

    def residual(self, state):
        '''
        计算残差
        :param measureData:  【np.array】观测值
        :return: 【np.array】观测值 - 预测值
        '''
        eastData = self.h(state)
        return self.measureData - eastData

    def derive(self, param_index):
        """
        指定状态量的偏导数
        :param param_index: 第几个状态量
        :return: 偏导数 (m, )
        """
        state1 = self.state.vector().copy()
        state2 = self.state.vector().copy()
        delta = 0.001
        state1[param_index] += delta
        state2[param_index] -= delta
        data_est_output1 = self.h(se3(vector=state1))
        data_est_output2 = self.h(se3(vector=state2))
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

    def LM(self, maxIter, printBool):
        """
        Levenberg–Marquardt优化算法的主体
        :param maxIter: 最大迭代次数
        :return: 【np.array】优化后的状态 (7, )
        """
        tao = 1e-3
        eps_stop = 1e-9
        eps_step = 1e-6
        eps_residual = 1e-3

        t0 = datetime.datetime.now()
        res = self.residual(self.state)
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
                    self.stateOut(t0, i, mse, 'threshold_step', printBool)
                    return self.state
                newState = se3(vector=self.state.vector() + step)   # 先将se3转换成数组相加，再变回李代数，这样才符合LM算法流程
                newRes = self.residual(newState)
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
                            self.stateOut(t0, i, mse, 'threshold_stop', printBool)
                        if mse <= eps_residual:
                            self.stateOut(t0, i, mse, 'threshold_residual', printBool)
                        return self.state
                    else:
                        break
                else:
                    u *= v
                    v *= 2

            self.stateOut(t0, i, mse, '', printBool)

    def stateOut(self, t0, i, mse, printStr, printBool):
        '''
        输出算法的中间结果
        :param t0: 【float】 时间戳
        :param i: 【int】迭代步数
        :param mse: 【float】残差
        :return:
        '''
        if not printBool:
            return
        print(printStr)
        timeCost = (datetime.datetime.now() - t0).total_seconds()
        pos = self.state.exp().matrix()[:3, 3]
        em = self.state.exp().matrix()[2, :3]
        pos = np.round(pos, 1)
        em = np.round(em, 3)
        print('i={}, pos={}mm, em={}, timeConsume={:.3f}s, cost={:.3e}'.format(i, pos, em, timeCost, mse))

    def sim(self):
        '''
        使用模拟的观测值验证算法的准确性
        :return:
        '''
        measureData = self.generateData(3)
        state = self.LM(100, True)

        posX = self.stateX.exp().matrix()[:3, 3]
        emX = self.stateX.exp().matrix()[2, :3]
        pos = self.state.exp().matrix()[:3, 3]
        em = self.state.exp().matrix()[2, :3]
        err_pos = np.linalg.norm(pos - posX) / np.linalg.norm(posX)
        err_em = np.linalg.norm(em - emX)      # 方向矢量本身是归一化的
        print('turth: pos={}, ez={}, err_pos={:.0%}, err_em={:.0%}'.format(np.round(posX, 1), np.round(emX, 3), err_pos, err_em))

if __name__ == '__main__':
    state = se3(vector=np.array([0, 0, 0, 0, 0, 100]))
    stateX = se3(vector=np.array([0.5, 0.2, 0.3, 0, 0, 233.7]))
    print(stateX.quaternion())
    currents = [2] * 16
    pred = Predictor(state, stateX, currents)
    pred.sim()
