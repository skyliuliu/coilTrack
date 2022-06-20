# coding=utf-8
# /usr/bin/env python3
"""
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2022/6/18 11:44
desc: 专用于定位的仿真类
"""
import numpy as np
import matplotlib.pyplot as plt

from predictorViewer import plotP, plotErr, plotTrajectory
from UKFpredictor import Tracker
from LMpredictor import Predictor


class Simulate(Tracker):
    plotBool = False
    printBool = False

    def __init__(self, currents, state0, plotType, sensor_std=None, sensor_err=None, ):
        """
        使用模拟的观测值验证算法的准确性
        :param currents:【float】发射线圈阵列的电流幅值[A]
        :param state0: 【np.array】初始值 (n,)
        :param sensor_std: 【float】sensor的噪声标准差[mG]
        :param sensor_err: 【float】sensor的噪声误差百分比[100%]
        :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'

        """
        super().__init__(currents, state0)
        self.sensor_std = sensor_std
        self.sensor_err = sensor_err
        self.plotType = plotType

    def simOne(self, stateX, maxIter=1):
        """
        仿真某一个点
        :param stateX: 【np.array】模拟的真实状态，可以有多个不同的状态
        :param maxIter: 【int】最大迭代次数
        :return: 【tuple】 位置[x, y, z]和姿态em2的误差
        """
        EX = self.h(stateX)
        Esim = np.zeros(self.m)
        for i in range(self.m):
            if self.sensor_err:
                Esim[i] = EX[i] * (1 + self.sensor_err * (-1) ** i)  # 百分比误差
                continue
            elif self.sensor_std:
                Esim[i] = np.random.normal(EX[i], self.sensor_std, 1)
                continue
            else:
                Esim[i] = EX[i]

        # 运行模拟数据
        for i in range(maxIter):
            if self.printBool:
                print('\n=========={}=========='.format(i))
            if self.plotBool:
                plt.ion()
                plotP(self, stateX, i, self.plotType)  # 画出卡曼协方差矩阵的椭圆图
                if i == maxIter - 1:  # 达到最大迭代书关闭交互模式
                    plt.ioff()
                    plt.show()
            posPre = self.state
            self.solve(Esim)
            delta_x = np.linalg.norm(self.state - posPre)
            # print('delta_x={:.3e}'.format(delta_x))

            if delta_x < 1e-3:  # 迭代步长的模长小于1e-3时关闭交互模式
                if self.plotBool:
                    plt.ioff()
                    plt.show()
                else:
                    break

        pos_em2 = self.parseState(self.state)
        pos, em2 = pos_em2[:3], pos_em2[3:]
        posX_em2X = self.parseState(stateX)
        posX, em2X = posX_em2X[:3], posX_em2X[3:]
        err_pos = np.linalg.norm(pos - posX)  # 位移之差的模
        err_em = np.arccos(np.dot(em2, em2X) / np.linalg.norm(em2) / np.linalg.norm(em2X)) * 57.3  # 方向矢量形成的夹角
        print('result: real_pos={}, err_pos={:.0f}mm, err_em={:.0f}\u00b0'.format(posX, err_pos, err_em))
        return err_pos, err_em

    def simErrDistributed(self, contourBar, pos_or_ori=0):
        """
        模拟误差分布
        :param contourBar: 【np.array】等高线的刻度条
        :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
        :return:
        """
        n = 20
        x, y = np.meshgrid(np.linspace(-200, 200, n), np.linspace(-200, 200, n))
        stateDist = np.array([0, 0, 200, np.pi / 4, -np.pi])
        z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                stateDist[0] = x[i, j]
                stateDist[1] = y[i, j]
                z[i, j] = self.simOne(stateX=stateDist)[pos_or_ori]

        plotErr(x, y, z, contourBar, titleName='sensor_err={},sensor_std={}'.format(self.sensor_err, self.sensor_std))

    def trajectorySim(self, shape, pointsNum):
        """
        模拟某条轨迹下的定位效果
        :param pointsNum: 【int】线上的点数
        :param shape: 【string】形状
        :return:
        实现流程
        1、定义轨迹
        2、提取轨迹上的点生成模拟数据
        3、提取预估的状态，并绘制预估轨迹
        """
        if shape == "straight":
            line = [[x, 100, 300] for x in np.linspace(-100, 100, pointsNum)]
        elif shape == "sin":
            line_x = np.linspace(-100, 100, pointsNum)
            line_y = np.sin(line_x / 50 * np.pi) * 100    # 固定两个周期：n=200/50/2=2
            line = [[x, y, 300] for (x, y) in zip(line_x, line_y)]
        elif shape == "circle":
            line0 = np.linspace(0, 2 * np.pi, pointsNum)
            line_x = np.sin(line0) * 100 + 100
            line_y = np.cos(line0) * 100
            line = [[x, y, 300] for (x, y) in zip(line_x, line_y)]
        else:
            raise TypeError("shape is not right!!!")

        stateLine = np.array([line[i] + list(self.state[3:]) for i in range(pointsNum)])
        state = stateLine[0]

        stateMP = []
        # 先对初始状态进行预估，给予足够的时间满足迭代误差内
        self.simOne(stateX=state, maxIter=30)
        stateMP.append(self.state.copy())

        # 对轨迹线上的其它点进行预估
        for i in range(1, pointsNum):
            print('\n--------point:{}---------'.format(i))
            # 固定迭代次数
            self.simOne(stateX=stateLine[i], maxIter=10)
            stateMP.append(self.state.copy())

        stateMP = np.asarray(stateMP)
        plotTrajectory(stateLine, stateMP, self.sensor_err)


if __name__ == '__main__':
    currents = [2] * 16
    # 球坐标系
    state0 = np.array([0, 0, 200, np.pi / 4, 0], dtype=float)
    stateX = np.array([0, -10, 200, np.pi / 4, np.pi / 4], dtype=float)
    # 四元数
    # state0 = np.array([0, 0, 200, 1, 0, 0, 0], dtype=float)  # 初始值
    # stateX = np.array([30, -10, 260, 1, 1, 2, 2], dtype=float)  # 真实值
    sim = Simulate(currents=[2] * 16, state0=state0, sensor_err=0.01, sensor_std=None, plotType=(1, 2))

    # sim.simOne(stateX, maxIter=50)

    # sim.simErrDistributed(contourBar=np.linspace(0, 50, 9))

    sim.trajectorySim(shape="circle", pointsNum=50)
