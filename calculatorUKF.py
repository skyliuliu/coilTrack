import datetime
import json
import time
from queue import Queue
from multiprocessing.dummy import Process

import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.stats import plot_covariance

from coilArray import CoilArray
from predictorViewer import q2R, plotP, plotErr, plotTrajectory, track3D
from readData import readRecData

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Tracker:

    deltaT = 10e-3  # 相邻发射线圈产生的接收信号的间隔时间[s]
    printBool = False   # 是否打印结果

    def __init__(self, currents, state0):
        '''
        :param currents:【float】发射线圈阵列的电流幅值[A]
        :param state0: 【np.array】初始值 (n,)
        '''
        self.n = state0.size  # 预测量：x, y, z, θ, φ
        self.dt = 0.5  # 时间间隔[s]
        self.currents = currents
        self.coils = CoilArray(np.array(currents))
        self.m = CoilArray.coilNum + 2
    
        points = MerweScaledSigmaPoints(n=self.n, alpha=0.3, beta=2., kappa=3 - self.n)
        self.ukf = UKF(dim_x=self.n, dim_z=self.m, dt=self.dt, points=points, fx=self.f, hx=self.hhh)
        self.ukf.x = state0.copy()  # 初始值
        self.state = self.ukf.x
        self.totalTime = 0
        self.compTime = 0
        self.t0 = time.time()
        self.iter = 1

        self.ukf.R *= 5
        self.ukf.P *= 10
        self.ukf.P[3, 3] = 1    # θ的初始协方差
        self.ukf.P[4, 4] = 0.1    # φ的初始协方差

        self.ukf.Q = np.eye(self.n) * 0.1 * self.dt  # 将速度作为过程噪声来源，Qi = v * dt
    
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
        '''
        静止状态下的转移函数，仅考虑速度为噪声项
        :param x: 上一时刻的状态
        :param dt: 时间间隔
        :return: 当前时刻的状态
        '''
        #A = np.eye(self.n)
        #return np.hstack(np.dot(A, x.reshape(self.n, 1)))
        return x


    # def f(self, x, dt):
    #     # 预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3 对应的转移函数
    #     A = np.eye(self.n)
    #     for i in range(0, 6, 2):
    #         A[i, i + 1] = dt
    #     return np.hstack(np.dot(A, x.reshape(self.n, 1)))

    def h(self, state):
        '''
        纯线圈的观测方程
        :param state: 预估的状态量 (n, )
        :return: 感应电压 [1e-6V] (m, )
        '''
        dArray0 = state[:3] - self.coils.coilArray
        #self.em2 = q2R(self.ukf.x[3: self.n])[:, -1]  # 四元数下的接收线圈朝向
        # 球坐标系下的接收线圈朝向
        theta, phi = state[3], state[4]
        em2 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]).T
        E = np.zeros(self.coils.coilNum)
        for i, d in enumerate(dArray0):
            # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em2)
            # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em2)
            # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
            E[i] = self.coils.inducedVolatage(d=d, em1=(0, 0, 1), em2=em2, ii=self.coils.currents[i])
        return E

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

    def hhh(self, state):
        '''
        基于θ和φ的线圈+IMU的观测方程
        :param state: 预估的状态量 (n, )
        :return: E+A 感应电压 [1e-6V] + 方向矢量[1] (m, )
        '''
        theta, phi = state[3], state[4]
        dArray0 = state[:3] - self.coils.coilArray
        # self.em2 = q2R(self.ukf.x[3: self.n])[:, -1]  # 四元数下的接收线圈朝向
        # 球坐标系下的接收线圈朝向
        theta, phi = state[3], state[4]
        em2 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]).T

        EZ = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            EZ[i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1, em2=em2, ii=self.currents[i])

       # 加速度的单位为100mg
        EZ[-1] = 10.24 * np.cos(theta)   # z
        EZ[-2] = 10.24 * np.sin(theta)   # sqrt(x^2 + y^2)

        return EZ

    def solve(self, z):
        '''
        根据观测值进行预测和更新
        :param z: 观测值 感应电压 [1e-6V]
        :return:
        '''
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

        if self.printBool:
            self.statePrint()

    def statePrint(self,):
        self.pos = np.round(self.ukf.x[:3], 3)
        # em2 = q2R(self.ukf.x[3: self.n])[:, -1]  # 四元数下的接收线圈朝向
        # 球坐标系下的接收线圈朝向
        theta, phi = self.ukf.x[3], self.ukf.x[4]
        em2 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]).T

        Estate = self.h(self.ukf.x)     # 计算每个状态对应的感应电压

        print('pos={}mm, emz={}, Emax={:.2f}, Emin={:.2f}, totalTime={:.3f}s'.format(
            self.pos, em2, max(abs(Estate)), min(abs(Estate)), self.totalTime))


def sim(sensor_std, sensor_err, plotType, state0, plotBool, printBool, state=None, maxIter=50):
    """
    使用模拟的观测值验证算法的准确性
    :param state: 【list】模拟的真实状态，可以有多个不同的状态
    :param sensor_std: 【float】sensor的噪声标准差[mG]
    :param sensor_err: 【float】sensor的噪声误差百分比[100%]
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    """
    if state is None:
        # state = [0, 0.2, 0.4, 0, 0, 1, 1]
        state = [0, 0.2, 0.4, 0.1, 0, 0, 0, 0, 1, 1]
    currents = [2] * CoilArray.coilNum
    tracker = Tracker(currents, state0)
    E = np.zeros(tracker.coils.coilNum)
    Esim = np.zeros((tracker.coils.coilNum, maxIter))
    useSaved = False

    if useSaved:
        f = open('Esim.json', 'r')
        simData = json.load(f)
        for j in range(tracker.coils.coilNum):
            for k in range(maxIter):
                Esim[j, k] = simData.get('Esim{}-{}'.format(j, k), 0)
        print('++++read saved Esim data+++')
    else:
        std = sensor_std
        # em1Sim = q2R(state[3: 7])[:, -1]
        # dArray = state[:3] - tracker.coilArray
        # for i, d in enumerate(dArray):
        #     # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em1Sim)  # x线圈阵列产生的感应电压中间值
        #     # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em1Sim)  # y线圈阵列产生的感应电压中间值
        #     # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em1Sim)  # z线圈阵列产生的感应电压中间值
        #     E[i] = inducedVolatage(d=d, em2=em1Sim)  # 单向线圈阵列产生的感应电压中间值
        E = tracker.h(state)

        simData = {}
        for j in range(tracker.coils.coilNum):
            #Esim[j, :] = np.random.normal(E[j], std, maxIter)
            Esim[j, :] = E[j] * (1 + sensor_err * (-1) ** j)
            # plt.hist(Esim[j, :], bins=25, histtype='bar', rwidth=2)
            # plt.show()
            for k in range(maxIter):
                simData['Esim{}-{}'.format(j, k)] = Esim[j, k]
        # 保存模拟数据到本地
        # f = open('Esim.json', 'w')
        # json.dump(simData, f, indent=4)
        # f.close()
        # print('++++save new Esim data+++')

    # 运行模拟数据
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
        tracker.solve(printBool, Esim[:, i])
        delta_x = np.linalg.norm(tracker.ukf.x[:3] - posPre)
        # print('delta_x={:.3e}'.format(delta_x))

        if delta_x < 1e-3:
            if plotBool:
                plt.ioff()
                plt.show()
            else:
                break

    err_pos = np.linalg.norm(tracker.ukf.x[:3] - state[:3]) / np.linalg.norm(state[:3])
    err_em = np.linalg.norm(q2R(tracker.ukf.x[3: 7])[:, -1] - q2R(state[3: 7])[:, -1])
    print('\nerr_std: pos={}, err_pos={:.0%}, err_em={:.0%}'.format(np.round(state[:3], 3), err_pos, err_em))
    return (err_pos, err_em)

def measureDataPredictor(sensor_std, state0, plotType, plotBool, printBool, state=None, maxIter=50):
    """
    使用实测结果估计位姿
    :param sensor_std: 【float】sensor的噪声标准差[1e-6V]
    :param state: 【list】模拟的真实状态，可以有多个不同的状态
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    """
    measureData = np.zeros(CoilArray.coilNum)
    with open('measureData.csv' , 'r', encoding='utf-8') as f:
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
        tracker.solve(printBool, measureData)
        delta_x = np.linalg.norm(tracker.ukf.x[:3] - posPre)
        # print('delta_x={:.3e}'.format(delta_x))

        if delta_x < 1e-4:
            if plotBool:
                plt.ioff()
                plt.show()
            else:
                break

def simErrDistributed(contourBar, sensor_std=10, pos_or_ori=1):
    """
    模拟误差分布
    :param contourBar: 【np.array】等高线的刻度条
    :param sensor_std: 【float】sensor的噪声标准差[mG]
    :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
    :return:
    """
    n = 20
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, n), np.linspace(-0.2, 0.2, n))
    state0Dist = np.array([0, 0, 0.3, 1, 0, 0, 0])
    stateDist = np.array([0, 0, 0.3, 1, 0, 0, 0])
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            stateDist[0] = x[i, j]
            stateDist[1] = y[i, j]
            z[i, j] = sim(sensor_std, plotType=(0, 1), state0=state0Dist, plotBool=False, printBool=False, state=stateDist)[pos_or_ori]

    plotErr(x, y, z, contourBar, titleName='sensor_std={}'.format(sensor_std))

def trajectorySim(shape, pointsNum, sensor_std, state0, plotBool, printBool, maxIter=50):
    '''
    模拟某条轨迹下的定位效果
    :param sensor_std: 【float】传感器噪声，此处指感应电压的采样噪声[μV]
    :param state0: 【np.array】初始状态 (7, )
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印日志
    :param maxIter: 【int】最大迭代次数
    :return:
    实现流程
    1、定义轨迹
    2、提取轨迹上的点生成模拟数据
    3、提取预估的状态，并绘制预估轨迹
    '''
    line = trajectoryLine(shape, pointsNum)
    q = [0, 0, 1, 1]
    stateLine = np.array([line[i] + q for i in range(pointsNum)])
    state =line[0] + q

    tracker = Tracker(sensor_std, state0, state)   # 创建UKF定位的对象
    stateMP = []

    # 先对初始状态进行预估，给予足够的时间满足迭代误差内
    E0sim = generateEsim(state, sensor_std, maxNum=maxIter)
    for i in range(maxIter):
        posPre = tracker.ukf.x[:3]
        tracker.solve(printBool, E0sim[:, i])

        delta_x = np.linalg.norm(tracker.ukf.x[:3] - posPre)
        if delta_x < 1e-2:
            print('=========state0 predict over============')
            break
    stateMP.append(tracker.ukf.x)

    # 对轨迹线上的其它点进行预估
    for i in range(1, pointsNum):
        print('--------point:{}---------'.format(i))
        state = line[i] + q
        '''
        # 以位置迭代的步长来判断是否收敛
        posPre = state0[:3]
        delta_x = np.linalg.norm(tracker.ukf.x[:3] - posPre)
        while delta_x > 0.004:
            Esim = generateEsim(state, sensor_std, maxNum=1)
            tracker.solve(printBool, Esim[:, 0])
            delta_x = np.linalg.norm(tracker.ukf.x[:3] - posPre)
            posPre = tracker.ukf.x[:3]
        '''
        # 固定迭代次数
        N = 10
        Esim = generateEsim(state, sensor_std, maxNum=N)
        for j in range(N):
            tracker.solve(printBool, Esim[:, j])
        stateMP.append(tracker.ukf.x)

    if plotBool:
        stateMP = np.asarray(stateMP)
        plotTrajectory(stateLine, stateMP, sensor_std)

def trajectoryLine(shape, pointsNum):
    '''
    生成指定形状的轨迹线
    :param shape: 【string】形状
    :param pointsNum: 【int】线上的点数
    :return: 【np.array】线上点的坐标集合 (pointsNum, 3)
    '''
    if shape == "straight":
        line = [[x, 0, 300] for x in np.linspace(-100, 100, pointsNum)]
    elif shape == "sin":
        line_x = np.linspace(-100, 100, pointsNum)
        line_y = np.sin(line_x * pointsNum * np.pi * 0.5) * 0.1
        line = [[x, y, 300] for (x, y) in zip(line_x, line_y)]
    elif shape == "circle":
        line0 = np.linspace(0, 2 * np.pi, pointsNum)
        line_x = np.sin(line0) * 100 + 100
        line_y = np.cos(line0) * 100
        line = [[x, y, 300] for (x, y) in zip(line_x, line_y)]
    else:
        raise TypeError("shape is not right!!!")
    return line

def generateEsim(state, sensor_std, maxNum=50):
    '''
    根据胶囊的状态给出接收线圈的感应电压模拟值
    :param state: 【np.array】 胶囊的真实状态 (7, )
    :param maxNum: 【int】最大数据量
    :param sensor_std: 【float】传感器噪声，此处指感应电压的采样噪声[μV]
    :return: 【np.array】感应电压模拟值 (Tracker.coils.coilNum, maxNum)
    '''
    currents = [1] * 16
    coils = CoilArray(currents)
    E = np.zeros(coils.coilNum)
    Esim = np.zeros((coils.coilNum, maxNum))
    em2Sim = q2R(state[3: 7])[:, -1]
    dArray = state[:3] - coils.coilArray
    for i, d in enumerate(dArray):
        E[i] = coils.inducedVolatage(d=d, em2=em2Sim, ii=currents[i])  # 单向线圈阵列产生的感应电压中间值

    for j in range(CoilArray.coilNum):
        Esim[j, :] = np.random.normal(E[j], sensor_std, maxNum)

    return Esim


def runReal():
    '''
    启动实时定位
    :return:
    '''
    qADC, qGyro, qAcc = Queue(), Queue(), Queue()
    state = np.array([0, 0, 200, np.pi / 4, 0], dtype=float)

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

