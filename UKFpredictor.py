import time
from queue import Queue
from multiprocessing.dummy import Process

import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from coilArray import CoilArray
from predictorViewer import q2R, plotP, track3D
from readData import readRecData

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Tracker:
    deltaT = 10e-3  # 相邻发射线圈产生的接收信号的间隔时间[s]
    printBool = False  # 是否打印结果

    def __init__(self, currents, state0):
        """
        :param currents:【float】发射线圈阵列的电流幅值[A]
        :param state0: 【np.array】初始值 (n,)
        """
        self.n = state0.size  # 预测量：x, y, z, θ, φ
        self.dt = 0.5  # 时间间隔[s]
        self.currents = currents
        self.coils = CoilArray(np.array(currents))
        self.m = CoilArray.coilNum + 2

        points = MerweScaledSigmaPoints(n=self.n, alpha=0.3, beta=2., kappa=3 - self.n)
        self.ukf = UKF(dim_x=self.n, dim_z=self.m, dt=self.dt, points=points, fx=self.f, hx=self.h)
        self.ukf.x = state0.copy()  # 初始值
        self.state = self.ukf.x
        self.totalTime = 0
        self.compTime = 0
        self.t0 = time.time()
        self.iter = 1

        self.ukf.R *= 5  # 观测值的协方差

        self.ukf.P *= 1000  # 初始位置x,y,z的协方差
        self.ukf.P[3, 3] = 1  # θ的初始协方差
        self.ukf.P[4, 4] = 0.4  # φ的初始协方差

        self.ukf.Q = np.eye(self.n) * 1 * self.dt  # 将速度作为过程噪声来源，Qi = v * dt
        self.ukf.Q[3, 3] = 0.5 * self.dt  # θ轴转动噪声
        self.ukf.Q[4, 4] = 0.5 * self.dt  # φ轴转动噪声

        self.pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))

        # 球坐标系下的接收线圈朝向
        theta, phi = state0[3], state0[4]
        self.em2 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]).T

    # def __init__(self, sensor_std, state0, state):
    #     '''
    #     预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3
    #     :param sensor_std:【float】sensor的噪声标准差[μV]
    #     :param state0: 【np.array】初始值 (10,)
    #     :param state: 【np.array】预测值 (10,)
    #     '''
    #     self.n = 10
    #     self.dt = 0.01  # 时间间隔[s]

    #     self.points = MerweScaledSigmaPoints(n=self.n, alpha=0.3, beta=2., kappa=3 - self.n)
    #     self.ukf = UKF(dim_x=self.n, dim_z=self.coils.coilNum, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
    #     self.ukf.x = state0.copy()  # 初始值
    #     self.x0 = state  # 计算NEES的真实值

    #     self.ukf.R *= sensor_std
    #     self.ukf.P = np.eye(self.n) * 0.01
    #     for i in range(6, 10):
    #         self.ukf.P[i, i] = 0.01

    #     self.ukf.Q = np.zeros((self.n, self.n))  # 初始化过程误差，全部置为零
    #     # 以加速度作为误差来源
    #     self.ukf.Q[0: 6, 0: 6] = Q_discrete_white_noise(dim=2, dt=self.dt, var=5, block_size=3)
    #     for i in range(6, 10):   # 四元数的过程误差
    #         self.ukf.Q[i, i] = 0.05

    #     self.pos = (round(self.ukf.x[0], 3), round(self.ukf.x[2], 3), round(self.ukf.x[4], 3))
    #     self.vel = (round(self.ukf.x[1], 3), round(self.ukf.x[3], 3), round(self.ukf.x[5], 3))
    #     self.em2 = q2R(self.ukf.x[6: 10])[:, -1]

    def f(self, x, dt):
        """
        静止状态下的转移函数，仅考虑速度为噪声项
        :param dt: 【float】时间间隔
        :param x: 上一时刻的状态
        :return: 当前时刻的状态
        """
        return x

    # def f(self, x, dt):
    #     # 预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3 对应的转移函数
    #     A = np.eye(self.n)
    #     for i in range(0, 6, 2):
    #         A[i, i + 1] = dt
    #     return np.hstack(np.dot(A, x.reshape(self.n, 1)))

    def h(self, state):
        """
        观测方程
        :param state: 预估的状态量 (n, )
        :return: 感应电压 [1e-6V] (m, )
        """
        pos, em2 = self.parseState(state)
        dArray0 = pos - self.coils.coilArray

        # 球坐标系下的接收线圈朝向
        EA = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em2)
            # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em2)
            # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
            EA[i] = self.coils.inducedVolatage(d=d, em1=(0, 0, 1), em2=em2, ii=self.coils.currents[i])
        if self.m == CoilArray.coilNum:  # 纯线圈
            return EA
        elif self.m == CoilArray.coilNum + 2:  # 基于θ和φ的线圈+IMU
            theta, phi = state[3], state[4]
            EA[-1] = 10.24 * np.cos(theta)  # z
            EA[-2] = 10.24 * np.sin(theta)  # sqrt(x^2 + y^2)
            return EA
        else:
            raise ValueError("观测量输入错误")

    # def h(self, state):
    #     # 预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3 对应的观测函数
    #     posNow = state[:6:2]   # 当前时刻的预测位置
    #     velNow = state[1:6:2]    # 当前时刻的预测速度
    #     em2 = np.array(q2R(state[6: 10]))[:, -1]
    #     E = np.zeros(self.coils.coilNum)
    #     for i in range(self.coilrows * self.coilcols):
    #         # 倒退第i个线圈的位移di，发射线圈到接收线圈的位置矢量为di - coilArray[i]
    #         di = posNow - velNow * (self.coilrows * self.coilcols - 1 - i) * self.deltaT - self.coilArray[i]
    #         E[i] = inducedVolatage(d=di, em1=(0, 0, 1), em2=em2)
    #     return E

    def solve(self, z):
        """
        根据观测值进行预测和更新
        :param z: 观测值 感应电压 [1e-6V]
        :return:
        """
        zz = np.hstack(z[:])
        # 附上时间戳
        t0 = time.time()
        # 开始预测和更新
        self.ukf.predict()
        self.ukf.update(zz)
        self.state = self.ukf.x
        self.totalTime = time.time() - self.t0
        self.compTime = time.time() - t0
        self.t0 = time.time()

        self.statePrint()

    def parseState(self, state):
        """
        从状态量中提取目标的位置和朝向
        :return: 【np.array】位置和朝向 (6,)
        """
        if self.n == 5:
            pos = state[:3]
            theta, phi = state[3], state[4]
            em2 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]).T
        elif self.n == 6:
            pos = state.exp().matrix()[:3, 3]
            em2 = state.exp().matrix()[:3, 2]
        elif self.n == 7:
            pos = state[:3]
            em2 = q2R(state[3: 7])[:, 2]
        else:
            raise ValueError("状态量输入错误")
        return pos, em2

    def statePrint(self):
        pos, em2 = self.parseState(self.ukf.x)
        Estate = self.h(self.ukf.x)  # 计算每个状态对应的感应电压

        if self.printBool:
            print('pos={}mm, emz={}, Emax={:.2f}, Emin={:.2f}, totalTime={:.3f}s'.format(
                pos, em2, max(abs(Estate)), min(abs(Estate)), self.totalTime))


def measureDataPredictor(state0, plotType, plotBool, printBool, state=None, maxIter=50):
    """
    使用实测结果估计位姿
    :param state0: 【np.array】初始值 (n,)
    :param state: 【list】模拟的真实状态，可以有多个不同的状态
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    """
    measureData = np.zeros(CoilArray.coilNum)
    with open('measureData.csv', 'r', encoding='utf-8') as f:
        readData = f.readlines()
    for i in range(CoilArray.coilNum):
        measureData[i] = eval(readData[i])

    if state is None:
        state = [0, 0, 0.3, 1, 0, 0, 0]
    currents = [2] * CoilArray.coilNum
    tracker = Tracker(currents, state0)

    for i in range(maxIter):
        if printBool:
            print('=========={}=========='.format(i))
        if plotBool:
            plt.ion()
            plotP(tracker, state, i, plotType)
            if i == maxIter - 1:
                plt.ioff()
                plt.show()
        posPre = tracker.ukf.x[:3]
        tracker.solve(measureData)
        delta_x = np.linalg.norm(tracker.ukf.x[:3] - posPre)
        # print('delta_x={:.3e}'.format(delta_x))

        if delta_x < 1e-4:
            if plotBool:
                plt.ioff()
                plt.show()
            else:
                break


def runReal():
    """
    启动实时定位
    :return:
    """
    qADC, qGyro, qAcc = Queue(), Queue(), Queue()
    state = np.array([0, 0, 200, np.pi / 4, -np.pi], dtype=float)

    # 读取接收端数据
    procReadRec = Process(target=readRecData, args=(qADC, qGyro, qAcc))
    procReadRec.daemon = True
    procReadRec.start()
    time.sleep(0.5)

    # 读取发射端的电流，然后创建定位器对象
    currents = [2.22, 2.2, 2.31, 2.37, 2.32, 2.26, 2.26, 2.37, 2.24, 2.37, 2.36, 2.32, 2.34, 2.42, 2.41, 2.3]
    # runsend(open=True)
    tracker = Tracker(currents, state)

    # 描绘3D轨迹
    track3D(state, qList=[qADC, qGyro, qAcc], tracker=tracker)


if __name__ == '__main__':
    runReal()
