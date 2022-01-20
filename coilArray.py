# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2022/1/11 16:40
desc: 发射线圈阵列
'''
import numpy as np



class CoilArray:
    distance = 0.1  # 初级线圈之间的距离[m]
    coilrows = 4    # 行数
    coilcols = 4    # 列数
    coilNum = coilrows * coilcols
    CAlength = distance * (coilrows - 1)   # 阵列的长度（x方向尺寸）
    CAwidth = distance * (coilcols - 1)    # 阵列的宽度（y方向尺寸）
    coilArray = np.zeros((coilrows * coilcols, 3))   # 描述线圈阵列在XY平面的坐标

    n1 = 205  # 发射线圈匝数
    nr1 = 9   # 发射线圈层数
    r1 = 5    # 发射线圈内半径【mm】
    d1 = 0.6  # 发射线圈线径【mm】

    n2 = 100   # 接收线圈匝数
    nr2 = 2    # 接收线圈层数
    r2 = 2.5   # 接收线圈内半径【mm】
    d2 = 0.05  # 接收线圈线径【mm】
    freq = 5000   # 工作频率【Hz】
    em1 = np.array([0, 0, 1], dtype=float)   # 发射线圈朝向

    def __init__(self, currents):
        for row in range(self.coilrows):
            for col in range(self.coilcols):
                self.coilArray[row * self.coilrows + col] = np.array(
                    [-0.5 * self.CAlength + self.distance * col, 0.5 * self.CAwidth - self.distance * row, 0])

        # 精确计算线圈的面积【m^2】，第i层线圈的面积为pi * (r + d * i) **2
        self.S1 = self.n1 // self.nr1 * np.pi * sum([(self.r1 + self.d1 * j) ** 2 for j in range(self.nr1)]) / 1000000
        self.S2 = self.n2 // self.nr2 * np.pi * sum([(self.r2 + self.d2 * k) ** 2 for k in range(self.nr2)]) / 1000000

        self.currents = currents

    def inducedVolatage(self, em2, ii, d):
        '''
        计算发射线圈在接收线圈中产生的感应电动势
        **************************
        *假设：                   *
        *1、线圈均可等效为磁偶极矩   *
        *2、线圈之间静止           *
        **************************
        :param em2: 接收线圈的朝向 【np.array (3, )】
        :param i: 接收线圈的电流[A]
        :param d: 接收线圈的坐标 【np.array (3, )】
        :return: 感应电压 [1e-6V]
        '''
        dNorm = np.linalg.norm(d)
        er = d / dNorm

        self.em1 /= np.linalg.norm(self.em1)
        em2 /= np.linalg.norm(em2)

        E = 2 * np.pi * 0.1 * self.freq * ii * self.S1 * self.S2 / dNorm ** 3 * (
                3 * np.dot(er, self.em1) * np.dot(er, em2) - np.dot(self.em1, em2))
        return abs(E)  # 单位1e-6V

    def solenoid(self, em2, ii, d):
        """
        基于毕奥-萨法尔定律，计算发射线圈在接收线圈中产生的感应电动势
        :param ii: 激励电流的幅值 [A]
        :param d: 初级线圈中心到次级线圈中心的位置矢量 [m]
        :param em2: 接收线圈的朝向
        :return E: 感应电压 [1e-6V]
        """
        nh = int(self.n1 / self.nr1)
        ntheta = 100
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        r = np.linspace(self.r1, self.r1 + self.nr1 * self.d1, self.nr1, endpoint=False)
        h = np.linspace(0, nh * self.d1, nh, endpoint=False)
        hh = np.array([[0, 0, hi] for hi in h])

        drxy = np.stack((np.cos(theta), np.sin(theta), np.zeros(ntheta)), 1)  # 电流元在xy平面的位置方向
        dlxy = np.stack((-np.sin(theta), np.cos(theta), np.zeros(ntheta)), 1)  # 电流元在xy平面的电流方向
        dlxy = np.vstack([dlxy] * nh)

        dr = np.zeros((ntheta * self.n1, 3), dtype=np.float)
        dl = np.zeros((ntheta * self.n1, 3), dtype=np.float)
        for i in range(self.nr1):
            dl[ntheta * nh * i: ntheta * nh * (i + 1), :] = r[i] * 2 * np.pi / ntheta * dlxy
            for j in range(nh):
                dr[ntheta * (i * nh + j): ntheta * (i * nh + j + 1), :] = r[i] * drxy + hh[j]

        er = d * 1000 - dr
        rNorm = np.linalg.norm(er, axis=1, keepdims=True)
        er0 = er / rNorm
        dB = 1e-4 * ii * np.cross(dl, er0) / rNorm ** 2
        B = dB.sum(axis=0)

        E = 2 * np.pi * self.freq * self.S2 * np.dot(B, em2) * 1e6
        return E


if __name__ == '__main__':
    c = CoilArray()
    em2 = np.array([0, 0, 1], dtype=float)
    ii = 2
    d = np.array([0, 0, 0.2 - 0.0075])
    E1 = c.inducedVolatage(em2, ii, d)
    E2 = c.solenoid(em2, ii, d)
    print('E1={:.0f}uV, E2={:.0f}uV'.format(E1, E2))
