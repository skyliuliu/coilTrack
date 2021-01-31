import datetime
import math
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.stats import plot_covariance
from scipy import linalg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def inducedVolatage(n1=200, nr1=8, n2=100, nr2=2, r1=5, d1=0.6, r2=2.5, d2=0.05, i=2, freq=20000, d=(0, 0, 0.2),
                    em1=(0, 0, 1), em2=(0, 0, 1)):
    """
    计算发射线圈在接收线圈中产生的感应电动势
    **************************
    *假设：                   *
    *1、线圈均可等效为磁偶极矩  *
    *2、线圈之间静止           *
    **************************
    :param n1: 发射线圈匝数 [1]
    :param nr1: 发射线圈层数 [1]
    :param n2: 接收线圈匝数 [1]
    :param nr2: 接收线圈层数 [1]
    :param r1: 发射线圈内半径 [mm]
    :param d1: 发射线圈线径 [mm]
    :param r2: 接收线圈内半径 [mm]
    :param d2: 接收线圈线径 [mm]
    :param i: 激励电流的幅值 [A]
    :param freq: 激励信号的频率 [Hz]
    :param d: 初级线圈中心到次级线圈中心的位置矢量 [m]
    :param em1: 发射线圈的朝向 [1]
    :param em2: 接收线圈的朝向 [1]
    :return E: 感应电压 [1e-6V]
    """
    dNorm = np.linalg.norm(d)
    er = d / dNorm

    em1 /= np.linalg.norm(em1)
    em2 /= np.linalg.norm(em2)

    # 精确计算线圈的面积，第i层线圈的面积为pi * (r + d * i) **2
    S1 = n1 // nr1 * math.pi * sum([(r1 + d1 * j) ** 2 for j in range(nr1)]) / 1000000
    S2 = n2 // nr2 * math.pi * sum([(r2 + d2 * k) ** 2 for k in range(nr2)]) / 1000000

    B = np.divide(pow(10, -7) * i * S1 * (3 * np.dot(er, em1) * er - em1), dNorm ** 3)

    E = 2 * math.pi * pow(10, -7) * freq * i * S1 * S2 / dNorm ** 3 * (
            3 * np.dot(er, em1) * np.dot(er, em2) - np.dot(em1, em2))
    return E * 1000000  # 单位1e-6V


def solenoidB(n1=200, nr1=8, n2=100, nr2=2, r1=5, d1=0.6, r2=2.5, d2=0.05, ii=2, freq=20000, d=(0, 0, 200),
              em1=(0, 0, 1), em2=(0, 0, 1)):
    """
        计算发射线圈在空间任意一点产生的磁场
        :param n1: 发射线圈匝数 [1]
        :param nr1: 发射线圈层数 [1]
        :param n2: 接收线圈匝数 [1]
        :param nr2: 接收线圈层数 [1]
        :param r1: 发射线圈内半径 [mm]
        :param d1: 发射线圈线径 [mm]
        :param r2: 接收线圈内半径 [mm]
        :param d2: 接收线圈线径 [mm]
        :param ii: 激励电流的幅值 [A]
        :param freq: 激励信号的频率 [Hz]
        :param d: 初级线圈中心到次级线圈中心的位置矢量 [mm]
        :param em1: 发射线圈的朝向 [1]
        :param em2: 接收线圈的朝向 [1]
        :return E: 感应电压 [1e-6V]
        """
    nh = int(n1 / nr1)
    ntheta = 100
    theta = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
    r = np.linspace(r1, r1 + nr1 * d1, nr1, endpoint=False)
    h = np.linspace(0, nh * d1, nh, endpoint=False)
    hh = np.array([[0, 0, hi] for hi in h])

    drxy = np.array([[math.cos(th), math.sin(th), 0] for th in theta])  # 电流元在xy平面的位置方向
    dlxy = np.array([np.array([-math.sin(th), math.cos(th), 0]) for th in theta])  # 电流元在xy平面的电流方向

    dr = np.zeros((ntheta * n1, 3), dtype=np.float)
    dl = np.zeros((ntheta * n1, 3), dtype=np.float)
    for i in range(nr1):
        for j in range(nh):
            dr[ntheta * (i * nh + j): ntheta * (i * nh + j + 1), :] = r[i] * drxy + hh[j]
            dl[ntheta * (i * nh + j): ntheta * (i * nh + j + 1), :] = r[i] * 2 * math.pi / ntheta * dlxy

    er = d - dr
    rNorm = np.linalg.norm(er, axis=1, keepdims=True)
    er0 = er / rNorm
    dB = 1e-4 * ii * np.cross(dl, er0) / rNorm ** 2
    B = np.array([sum(dB[:, i]) for i in range(3)])
    return B


def q2m(q0, q1, q2, q3):
    qq2 = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    mx = 2 * (q0 * q2 + q1 * q3) / qq2
    my = 2 * (-q0 * q1 + q2 * q3) / qq2
    mz = (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) / qq2
    return [round(mx, 3), round(my, 3), round(mz, 3)]


class Tracker:
    distance = 0.15  # 初级线圈之间的距离[m]
    coilrows = 4
    coilcols = 4
    CAlength = distance * (coilrows - 1)
    CAwidth = distance * (coilcols - 1)
    coilArray = np.zeros((coilrows * coilcols, 3))
    for row in range(coilrows):
        for col in range(coilcols):
            coilArray[row * coilrows + col] = np.array(
                [-0.5 * CAlength + distance * col, 0.5 * CAwidth - distance * row, 0])

    def __init__(self, x0):
        self.stateNum = 7  # 预测量：x, y, z, q0, q1, q2, q3
        self.measureNum = self.coilrows * self.coilcols * 1
        self.dt = 0.01  # 时间间隔[s]

        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3 - self.stateNum)
        self.ukf = UKF(dim_x=self.stateNum, dim_z=self.measureNum, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = np.array([0, 0, 0.2, 1, 0, 0, 0])  # 初始值
        self.x0 = x0  # 计算NEES的真实值

        self.ukf.R *= 25
        self.ukf.P = np.eye(self.stateNum) * 0.01
        for i in range(3, 7):
            self.ukf.P[i, i] = 0.01
        self.ukf.Q = np.eye(self.stateNum) * 0.001 * self.dt  # 将速度作为过程噪声来源，Qi = [v*dt]
        for i in range(3, 7):
            self.ukf.Q[i, i] = 0.01  # 四元数的过程误差

        self.pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        self.m = q2m(self.ukf.x[3], self.ukf.x[4], self.ukf.x[5], self.ukf.x[6])

    def f(self, x, dt):
        A = np.eye(self.stateNum)
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def h(self, state):
        dArray0 = state[:3] - self.coilArray
        q0, q1, q2, q3 = state[3:7]
        em2 = np.array(q2m(q0, q1, q2, q3))
        E = np.zeros(self.measureNum)
        for i, d in enumerate(dArray0):
            # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em2)
            # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em2)
            # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
            E[i] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
        return E

    def run(self, Edata):
        # 输出预测结果
        print(r'pos={}m, e_moment={}'.format(self.pos, self.m))

        z = np.hstack(Edata[:])
        # 附上时间戳
        # t0 = datetime.datetime.now()
        # 开始预测和更新
        self.ukf.predict()
        self.ukf.update(z)
        # timeCost = (datetime.datetime.now() - t0).total_seconds()
        # state[:] = np.concatenate((self.ukf.x, np.array([timeCost])))  # 输出的结果

        Estate = self.h(self.ukf.x)
        print('Emax={:.2f}, Emin={:.2f}'.format(max(abs(Estate)), min(abs(Estate))))
        # 计算NEES值
        x = self.x0 - self.ukf.x
        nees = np.dot(x.T, linalg.inv(self.ukf.P)).dot(x)
        print('NEES={:.1f}'.format(nees))

        self.pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        self.m = q2m(self.ukf.x[3], self.ukf.x[4], self.ukf.x[5], self.ukf.x[6])

    def plotP(self, state, index):
        xtruth = state[:3]
        xtruth[1] += index  # 获取坐标真实值
        mtruth = q2m(state[3], state[4], state[5], state[6])  # 获取姿态真实值
        pos2 = np.zeros(2)
        pos2[0], pos2[1] = self.pos[1] + index, self.pos[2]  # 预测的坐标值
        Pxy = self.ukf.P[1:3, 1:3]  # 坐标的误差协方差
        plot_covariance(mean=pos2, cov=Pxy, fc='g', alpha=0.3, title='线圈定位过程仿真')
        plt.text(pos2[0], pos2[1], int(index * 10), fontsize=9)
        plt.plot(xtruth[1], xtruth[2], 'ro')  # 画出真实值
        plt.text(xtruth[1], xtruth[2], int(index * 10), fontsize=9)

        # 添加磁矩方向箭头
        plt.annotate(text='', xy=(pos2[0] + self.m[1] * 0.05, pos2[1] + self.m[2] * 0.05), xytext=(pos2[0], pos2[1]),
                     color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
        plt.annotate(text='', xy=(xtruth[1] + mtruth[1] * 0.05, xtruth[2] + mtruth[2] * 0.05),
                     xytext=(xtruth[1], xtruth[2]),
                     color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))
        # 添加坐标轴标识
        plt.xlabel('y/m')
        plt.ylabel('z/m')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().grid(b=True)
        plt.pause(0.05)


def sim(state=None):
    # 使用模拟的实测结果，测试UKF滤波器的参数设置是否合理
    if state is None:
        state = [0, 0.2, 0.4, 0, 0, 1, 1]
    mp = Tracker(state)
    nmax = 30  # 迭代次数
    E = np.zeros(mp.measureNum)
    Esim = np.zeros((mp.measureNum, nmax))
    useSaved = False

    if useSaved:
        f = open('Esim.json', 'r')
        simData = json.load(f)
        for j in range(mp.measureNum):
            for k in range(nmax):
                Esim[j, k] = simData.get('Esim{}-{}'.format(j, k), 0)
        print('++++read saved Esim data+++')
    else:
        std = 5
        em1Sim = q2m(*state[3:])
        dArray = state[:3] - mp.coilArray
        for i, d in enumerate(dArray):
            # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em1Sim)  # x线圈阵列产生的感应电压中间值
            # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em1Sim)  # y线圈阵列产生的感应电压中间值
            # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em1Sim)  # z线圈阵列产生的感应电压中间值
            E[i] = inducedVolatage(d=d, em2=em1Sim)  # 单向线圈阵列产生的感应电压中间值

        simData = {}
        for j in range(mp.measureNum):
            Esim[j, :] = np.random.normal(E[j], std, nmax)
            # plt.hist(Esim[j, :], bins=25, histtype='bar', rwidth=2)
            # plt.show()
            for k in range(nmax):
                simData['Esim{}-{}'.format(j, k)] = Esim[j, k]
        # 保存模拟数据到本地
        # f = open('Esim.json', 'w')
        # json.dump(simData, f, indent=4)
        # f.close()
        # print('++++save new Esim data+++')

    # 运行模拟数据
    n = 30
    for i in range(n):
        print('=========={}=========='.format(i))

        plt.ion()
        mp.plotP(state, i * 0.1)
        if i == n - 1:
            plt.ioff()
            plt.show()

        mp.run(Esim[:, i])
        time.sleep(0.05)


if __name__ == '__main__':
    # sim(state=[0, 0.2, 0.3, 0, 1, 0, 0])
    n = 9
    dzs = np.linspace(0.1, 0.4, n)
    B = solenoidB()
    E = inducedVolatage()
    print(B)
