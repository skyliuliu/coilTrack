# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2022/1/11 16:40
desc: 发射线圈阵列
'''
import numpy as np


def q2R(q):
    '''
    从四元数求旋转矩阵
    :param q: 四元数
    :return: R 旋转矩阵
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])
    return R


class CoilArray:
    distance = 100  # 初级线圈之间的距离[mm]
    coilRows = 4  # 行数
    coilCols = 4  # 列数
    coilNum = coilRows * coilCols
    CALength = distance * (coilRows - 1)  # 阵列的长度（x方向尺寸）
    CAWidth = distance * (coilCols - 1)  # 阵列的宽度（y方向尺寸）
    coilArray = np.zeros((coilRows * coilCols, 3))  # 描述线圈阵列在XY平面的坐标

    n1 = 205  # 发射线圈匝数
    nr1 = 9  # 发射线圈层数
    r1 = 5  # 发射线圈内半径【mm】
    d1 = 0.6  # 发射线圈线径【mm】

    n2 = 100  # 接收线圈匝数
    nr2 = 2  # 接收线圈层数
    r2 = 2.5  # 接收线圈内半径【mm】
    d2 = 0.05  # 接收线圈线径【mm】
    freq = 5000  # 工作频率【Hz】
    em1 = np.array([0, 0, 1], dtype=float)  # 发射线圈朝向
    em1x = np.array([1, 0, 0], dtype=float)  # 发射线圈x朝向
    em1y = np.array([0, 1, 0], dtype=float)  # 发射线圈y朝向
    em1z = np.array([0, 0, 1], dtype=float)  # 发射线圈z朝向

    def __init__(self, currents):
        """
        排列方式：
        0 1 2 3
        4 5 6 7
        8 9 10 11
        12 13 14 15
        """
        # XY坐标：
        for row in range(self.coilRows):
            for col in range(self.coilCols):
                self.coilArray[row * self.coilRows + col] = np.array(
                    [-0.5 * self.CALength + self.distance * col, 0.5 * self.CAWidth - self.distance * row, 0])

        # 线圈朝向
        angleInner = 45 * np.pi / 180  # 与z轴的夹角，向外旋转
        angleOuter = 45 * np.pi / 180  # 与z轴的夹角，向外旋转
        self.em1s = np.array([
            [-1, 1, np.sqrt(2) / np.tan(angleOuter)], [-1, 3, np.sqrt(10) / np.tan(angleOuter)],
            [1, 3, np.sqrt(10) / np.tan(angleOuter)], [1, 1, np.sqrt(2) / np.tan(angleOuter)],
            [-3, 1, np.sqrt(10) / np.tan(angleOuter)], [-1, 1, np.sqrt(2) / np.tan(angleInner)],
            [1, 1, np.sqrt(2) / np.tan(angleInner)], [3, 1, np.sqrt(10) / np.tan(angleOuter)],
            [-3, -1, np.sqrt(10) / np.tan(angleOuter)], [-1, -1, np.sqrt(2) / np.tan(angleInner)],
            [1, -1, np.sqrt(2) / np.tan(angleInner)], [3, -1, np.sqrt(10) / np.tan(angleOuter)],
            [-1, -1, np.sqrt(2) / np.tan(angleOuter)], [-1, -3, np.sqrt(10) / np.tan(angleOuter)],
            [1, -3, np.sqrt(10) / np.tan(angleOuter)], [1, -1, np.sqrt(2) / np.tan(angleOuter)]
        ])

        '''
        XYZ交错排列
        [0,0,1], [1,0,0], [0,-1,0],[0,0,1],
        [1,0,0], [0,1,0], [1,0,0], [0,-1,0], 
        [0,1,0], [-1,0,0], [0,-1,0], [-1,0,0], 
        [0,0,1], [0,1,0], [-1,0,0], [0,0,1]
        '''
        # self.em1s = np.array([
        #     [0,0,1], [1,0,0], [0,-1,0],[0,0,1],
        #     [1,0,0], [0,1,0], [1,0,0], [0,-1,0], 
        #     [0,1,0], [-1,0,0], [0,-1,0], [-1,0,0], 
        #     [0,0,1], [0,1,0], [-1,0,0], [0,0,1]
        # ], dtype=float)

        '''
        XYZ交错排列2
        X Y Z X
        Y Z X Y
        Z X Y Z
        X Y Z X
        '''
        self.em1s = np.array([
            [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0],
            [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]
        ], dtype=float)

        # 精确计算线圈的面积【mm^2】，第i层线圈的面积为pi * (r + d * i) **2
        self.S1 = self.n1 // self.nr1 * np.pi * sum([(self.r1 + self.d1 * j) ** 2 for j in range(self.nr1)])
        self.S2 = self.n2 // self.nr2 * np.pi * sum([(self.r2 + self.d2 * k) ** 2 for k in range(self.nr2)])

        self.currents = currents

    def inducedVolatage(self, em1, em2, ii, d):
        '''
        计算发射线圈在接收线圈中产生的感应电动势
        **************************
        *假设：                   *
        *1、线圈均可等效为磁偶极矩  *
        *2、线圈之间静止           *
        **************************
        :param em1: 发射线圈的朝向 【np.array (3, )】
        :param em2: 接收线圈的朝向 【np.array (3, )】
        :param i: 接收线圈的电流[A]
        :param d: 接收线圈的坐标 【np.array (3, )】
        :return: 感应电压 [1e-6V]
        '''
        dNorm = np.linalg.norm(d)
        er = d / dNorm

        em1 /= np.linalg.norm(em1)
        em2 /= np.linalg.norm(em2)

        E = 2 * np.pi * 0.1 * self.freq * ii * self.S1 * self.S2 / dNorm ** 3 * (
                3 * np.dot(er, em1) * np.dot(er, em2) - np.dot(em1, em2)) / 1000
        return abs(E)    # 实测结果中感应电压只能为正值

    def h(self, state):
        """
        纯线圈的观测方程
        :param state: 预估的状态量 (n, )
        :return: E 感应电压 [1e-6V] (m, )
        """
        dArray0 = state[:3] - self.coilArray
        em2 = q2R(state[3: 7])[:, -1]

        E = np.zeros(self.coilNum)
        for i, d in enumerate(dArray0):
            E[i] = self.inducedVolatage(d=d, em1=self.em1, em2=em2, ii=self.currents[i])
        return E

    def hh(self, state):
        """
        线圈+IMU的观测方程
        :param state: 预估的状态量 (n, )
        :return: E 感应电压 [1e-6V] (m, )
        """
        dArray0 = state[:3] - self.coilArray
        em2 = q2R(state[3: 7])[:, -1]

        EA = np.zeros(self.coilNum + 3)
        for i, d in enumerate(dArray0):
            EA[i] = self.inducedVolatage(d=d, em1=self.em1, em2=em2, ii=self.currents[i])

        EA[-3] = -1000 * em2[-3]  # x反向
        EA[-2] = -1000 * em2[-2]  # y反向
        EA[-1] = 1000 * em2[-1]  # z正向
        return EA


if __name__ == '__main__':
    currents = [2.21, 2.22, 2.31, 2.39, 2.33, 2.31, 2.29, 2.34, 2.29, 2.38, 2.36, 2.31, 2.35, 2.41, 2.42, 2.35]
    coils = CoilArray(np.array(currents))
    state = np.array([0, -5, 195 + 7.5, 1, 0, 0, 0])
    vm = coils.h(state)
    print('vm(uV):\n', np.round(vm, 0))

    A = 4 * np.pi * 1e-10 * coils.n1 * coils.n2 * coils.S1 * coils.S2 * coils.freq * 2
    print('A=', A)
