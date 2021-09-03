import datetime
import math
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.stats import plot_covariance

from predictorViewer import q2R, plotP, plotErr, plotTrajectory

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def inducedVolatage(n1=210, nr1=9, n2=100, nr2=2, r1=5, d1=0.6, r2=2.5, d2=0.05, i=0.96, freq=20000, d=(0, 0, 0.3 - 0.0075),
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
    :param d: 初级线圈中心到次级线圈中心的位置矢量 [m]，注意减去发射线圈的一半高度
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
    return abs(E) * 1000000  # 单位1e-6V


def solenoid(n1=210, nr1=9, n2=100, nr2=2, r1=5, d1=0.6, r2=2.5, d2=0.05, ii=0.86, freq=20000, d=(0, 0, 0.3 - 0.0075),
              em1=(0, 0, 1), em2=(0, 0, 1)):
    """
    基于毕奥-萨法尔定律，计算发射线圈在接收线圈中产生的感应电动势
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
    :param d: 初级线圈中心到次级线圈中心的位置矢量 [m]
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

    drxy = np.stack((np.cos(theta), np.sin(theta), np.zeros(ntheta)), 1)  # 电流元在xy平面的位置方向
    dlxy = np.stack((-np.sin(theta), np.cos(theta), np.zeros(ntheta)), 1)  # 电流元在xy平面的电流方向
    dlxy = np.vstack([dlxy] * nh)

    dr = np.zeros((ntheta * n1, 3), dtype=np.float)
    dl = np.zeros((ntheta * n1, 3), dtype=np.float)
    for i in range(nr1):
        dl[ntheta * nh * i: ntheta * nh * (i + 1), :] = r[i] * 2 * math.pi / ntheta * dlxy
        for j in range(nh):
            dr[ntheta * (i * nh + j): ntheta * (i * nh + j + 1), :] = r[i] * drxy + hh[j]

    er = np.array(d) * 1000 - dr
    rNorm = np.linalg.norm(er, axis=1, keepdims=True)
    er0 = er / rNorm
    dB = 1e-4 * ii * np.cross(dl, er0) / rNorm ** 2
    B = dB.sum(axis=0)

    # 精确计算线圈的面积，第i层线圈的面积为pi * (r + d * i) **2
    S2 = n2 // nr2 * math.pi * sum([(r2 + d2 * k) ** 2 for k in range(nr2)]) / 1000000
    E = 2 * math.pi * freq * S2 * np.dot(B, em2) * 1e6
    return E



class Tracker:
    distance = 0.10  # 初级线圈之间的距离[m]
    coilrows = 4
    coilcols = 4
    measureNum = coilrows * coilcols * 1
    CAlength = distance * (coilrows - 1)
    CAwidth = distance * (coilcols - 1)
    coilArray = np.zeros((coilrows * coilcols, 3))
    for row in range(coilrows):
        for col in range(coilcols):
            coilArray[row * coilrows + col] = np.array(
                [-0.5 * CAlength + distance * col, 0.5 * CAwidth - distance * row, 0])

    deltaT = 10e-3  # 相邻发射线圈产生的接收信号的间隔时间[s]

    def __init__(self, sensor_std, state0, state):
        '''
        预测量：x, y, z, q0, q1, q2, q3
        :param sensor_std:【float】sensor的噪声标准差[μV]
        :param state0: 【np.array】初始值 (7,)
        :param state: 【np.array】预测值 (7,)
        '''
        self.stateNum = 7  # 预测量：x, y, z, q0, q1, q2, q3
        self.dt = 0.01  # 时间间隔[s]
    
        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3 - self.stateNum)
        self.ukf = UKF(dim_x=self.stateNum, dim_z=self.measureNum, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = state0.copy()  # 初始值
        self.x0 = state  # 计算NEES的真实值
    
        self.ukf.R *= sensor_std
        self.ukf.P = np.eye(self.stateNum) * 0.01
        for i in range(3, 7):
            self.ukf.P[i, i] = 0.01
        self.ukf.Q = np.eye(self.stateNum) * 0.001 * self.dt  # 将速度作为过程噪声来源，Qi = [v*dt]
        for i in range(3, 7):
            self.ukf.Q[i, i] = 0.01  # 四元数的过程误差
    
        self.pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        self.m = q2R(self.ukf.x[3: 7])[:, -1]

    # def __init__(self, sensor_std, state0, state):
    #     '''
    #     预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3
    #     :param sensor_std:【float】sensor的噪声标准差[μV]
    #     :param state0: 【np.array】初始值 (10,)
    #     :param state: 【np.array】预测值 (10,)
    #     '''
    #     self.stateNum = 10
    #     self.dt = 0.01  # 时间间隔[s]

    #     self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3 - self.stateNum)
    #     self.ukf = UKF(dim_x=self.stateNum, dim_z=self.measureNum, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
    #     self.ukf.x = state0.copy()  # 初始值
    #     self.x0 = state  # 计算NEES的真实值

    #     self.ukf.R *= sensor_std
    #     self.ukf.P = np.eye(self.stateNum) * 0.01
    #     for i in range(6, 10):
    #         self.ukf.P[i, i] = 0.01

    #     self.ukf.Q = np.zeros((self.stateNum, self.stateNum))  # 初始化过程误差，全部置为零
    #     # 以加速度作为误差来源
    #     self.ukf.Q[0: 6, 0: 6] = Q_discrete_white_noise(dim=2, dt=self.dt, var=5, block_size=3)
    #     for i in range(6, 10):   # 四元数的过程误差
    #         self.ukf.Q[i, i] = 0.05

    #     self.pos = (round(self.ukf.x[0], 3), round(self.ukf.x[2], 3), round(self.ukf.x[4], 3))
    #     self.vel = (round(self.ukf.x[1], 3), round(self.ukf.x[3], 3), round(self.ukf.x[5], 3))
    #     self.m = q2R(self.ukf.x[6: 10])[:, -1]

    def f(self, x, dt):
        A = np.eye(self.stateNum)
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))


    # def f(self, x, dt):
    #     # 预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3 对应的转移函数
    #     A = np.eye(self.stateNum)
    #     for i in range(0, 6, 2):
    #         A[i, i + 1] = dt
    #     return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def h(self, state):
        dArray0 = state[:3] - self.coilArray
        em2 = np.array(q2R(state[3:7]))[:, -1]
        E = np.zeros(self.measureNum)
        for i, d in enumerate(dArray0):
            # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em2)
            # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em2)
            # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
            E[i] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
        return E

    # def h(self, state):
    #     # 预测量：x,vx, y, vz, z, vz, q0, q1, q2, q3 对应的观测函数
    #     posNow = state[:6:2]   # 当前时刻的预测位置
    #     velNow = state[1:6:2]    # 当前时刻的预测速度
    #     em2 = np.array(q2R(state[6: 10]))[:, -1]
    #     E = np.zeros(self.measureNum)
    #     for i in range(self.coilrows * self.coilcols):
    #         # 倒退第i个线圈的位移di，发射线圈到接收线圈的位置矢量为di - coilArray[i]
    #         di = posNow - velNow * (self.coilrows * self.coilcols - 1 - i) * self.deltaT - self.coilArray[i]
    #         E[i] = inducedVolatage(d=di, em1=(0, 0, 1), em2=em2)
    #     return E

    def run(self, printBool, Edata):
        z = np.hstack(Edata[:])
        # 附上时间戳
        self.t0 = datetime.datetime.now()
        # 开始预测和更新
        self.ukf.predict()
        self.ukf.update(z)

        if printBool:
            self.statePrint()

    def statePrint(self,):
        self.pos = np.round(self.ukf.x[:3], 3)
        self.m = np.round(q2R(self.ukf.x[3: 7])[:, -1], 3)

        timeCost = (datetime.datetime.now() - self.t0).total_seconds()
        Estate = self.h(self.ukf.x)     # 计算每个状态对应的感应电压
        # 计算NEES值
        x = self.x0 - self.ukf.x
        nees = np.dot(x.T, np.linalg.inv(self.ukf.P)).dot(x)
        print('pos={}m, emz={}, Emax={:.2f}, Emin={:.2f}, NEES={:.1f}, timeCost={:.3f}s'.format(
            self.pos, self.m, max(abs(Estate)), min(abs(Estate)), nees, timeCost))


def sim(sensor_std, plotType, state0, plotBool, printBool, state=None, maxIter=50):
    """
    使用模拟的观测值验证算法的准确性
    :param state: 【list】模拟的真实状态，可以有多个不同的状态
    :param sensor_std: 【float】sensor的噪声标准差[mG]
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    """
    if state is None:
        # state = [0, 0.2, 0.4, 0, 0, 1, 1]
        state = [0, 0.2, 0.4, 0.1, 0, 0, 0, 0, 1, 1]
    mp = Tracker(sensor_std, state0 ,state)
    E = np.zeros(mp.measureNum)
    Esim = np.zeros((mp.measureNum, maxIter))
    useSaved = False

    if useSaved:
        f = open('Esim.json', 'r')
        simData = json.load(f)
        for j in range(mp.measureNum):
            for k in range(maxIter):
                Esim[j, k] = simData.get('Esim{}-{}'.format(j, k), 0)
        print('++++read saved Esim data+++')
    else:
        std = sensor_std
        # em1Sim = q2R(state[3: 7])[:, -1]
        # dArray = state[:3] - mp.coilArray
        # for i, d in enumerate(dArray):
        #     # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em1Sim)  # x线圈阵列产生的感应电压中间值
        #     # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em1Sim)  # y线圈阵列产生的感应电压中间值
        #     # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em1Sim)  # z线圈阵列产生的感应电压中间值
        #     E[i] = inducedVolatage(d=d, em2=em1Sim)  # 单向线圈阵列产生的感应电压中间值
        E = mp.h(state)

        simData = {}
        for j in range(mp.measureNum):
            Esim[j, :] = np.random.normal(E[j], std, maxIter)
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
            plotP(mp, state, i, plotType)
            if i == maxIter - 1:
                plt.ioff()
                plt.show()
        posPre = mp.ukf.x[:3]
        mp.run(printBool, Esim[:, i])
        delta_x = np.linalg.norm(mp.ukf.x[:3] - posPre)
        # print('delta_x={:.3e}'.format(delta_x))

        if delta_x < 1e-3:
            if plotBool:
                plt.ioff()
                plt.show()
            else:
                break

    err_pos = np.linalg.norm(mp.ukf.x[:3] - state[:3]) / np.linalg.norm(state[:3])
    err_em = np.linalg.norm(q2R(mp.ukf.x[6: 10])[:, -1] - q2R(state[6: 10])[:, -1])
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
    measureData = np.zeros(Tracker.measureNum)
    with open('measureData.csv' , 'r', encoding='utf-8') as f:
        readData = f.readlines()
    for i in range(Tracker.measureNum):
        measureData[i] = eval(readData[i])

    if state is None:
        state = [0, 0, 0.3, 1, 0, 0, 0]
    mp = Tracker(sensor_std, state0 ,state)

    for i in range(maxIter):
        if printBool:
            print('=========={}=========='.format(i))
        if plotBool:
            plt.ion()
            plotP(mp, state, i, plotType)
            if i == maxIter - 1:
                plt.ioff()
                plt.show()
        posPre = mp.ukf.x[:3]
        mp.run(printBool, measureData)
        delta_x = np.linalg.norm(mp.ukf.x[:3] - posPre)
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

    mp = Tracker(sensor_std, state0, state)   # 创建UKF定位的对象
    stateMP = []

    # 先对初始状态进行预估，给予足够的时间满足迭代误差内
    E0sim = generateEsim(state, sensor_std, maxNum=maxIter)
    for i in range(maxIter):
        posPre = mp.ukf.x[:3]
        mp.run(printBool, E0sim[:, i])

        delta_x = np.linalg.norm(mp.ukf.x[:3] - posPre)
        if delta_x < 1e-2:
            print('=========state0 predict over============')
            break
    stateMP.append(mp.ukf.x)

    # 对轨迹线上的其它点进行预估
    for i in range(1, pointsNum):
        print('--------point:{}---------'.format(i))
        state = line[i] + q
        '''
        # 以位置迭代的步长来判断是否收敛
        posPre = state0[:3]
        delta_x = np.linalg.norm(mp.ukf.x[:3] - posPre)
        while delta_x > 0.004:
            Esim = generateEsim(state, sensor_std, maxNum=1)
            mp.run(printBool, Esim[:, 0])
            delta_x = np.linalg.norm(mp.ukf.x[:3] - posPre)
            posPre = mp.ukf.x[:3]
        '''
        # 固定迭代次数
        N = 10
        Esim = generateEsim(state, sensor_std, maxNum=N)
        for j in range(N):
            mp.run(printBool, Esim[:, j])
        stateMP.append(mp.ukf.x)

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
        line = [[x, 0, 0.3] for x in np.linspace(-0.1, 0.1, pointsNum)]
    elif shape == "sin":
        line_x = np.linspace(-0.1, 0.1, pointsNum)
        line_y = np.sin(line_x * pointsNum * math.pi * 0.5) * 0.1
        line = [[x, y, 0.3] for (x, y) in zip(line_x, line_y)]
    elif shape == "circle":
        line0 = np.linspace(0, 2 * math.pi, pointsNum)
        line_x = np.sin(line0) * 0.1 + 0.1
        line_y = np.cos(line0) * 0.1
        line = [[x, y, 0.3] for (x, y) in zip(line_x, line_y)]
    else:
        raise TypeError("shape is not right!!!")
    return line

def generateEsim(state, sensor_std, maxNum=50):
    '''
    根据胶囊的状态给出接收线圈的感应电压模拟值
    :param state: 【np.array】 胶囊的真实状态 (7, )
    :param maxNum: 【int】最大数据量
    :param sensor_std: 【float】传感器噪声，此处指感应电压的采样噪声[μV]
    :return: 【np.array】感应电压模拟值 (Tracker.measureNum, maxNum)
    '''
    E = np.zeros(Tracker.measureNum)
    Esim = np.zeros((Tracker.measureNum, maxNum))
    em2Sim = q2R(state[3: 7])[:, -1]
    dArray = state[:3] - Tracker.coilArray
    for i, d in enumerate(dArray):
        E[i] = inducedVolatage(d=d, em2=em2Sim)  # 单向线圈阵列产生的感应电压中间值

    for j in range(Tracker.measureNum):
        Esim[j, :] = np.random.normal(E[j], sensor_std, maxNum)

    return Esim

if __name__ == '__main__':
    state0 = np.array([0, 0, 0.3, 1, 0, 0, 0])
    state = np.array([-0.025, -0.025, 0.246, 0.5 * math.sqrt(3), 0.5, 0, 0])

    # state0 = np.array([0, 0, 0, 0, 0.3, 0, 1, 0, 0, 0])   # x,vx, y, vz, z, vz, q0, q1, q2, q3
    # state = np.array([0.16, 0.5, 0.2, 0, 0.3, 0, 1, 0, 0, 0])
    # sim(sensor_std=5, state0=state0, state=state, plotBool=False, printBool=True, plotType=(1, 2))

    # simErrDistributed(contourBar=np.linspace(0, 0.5, 9), sensor_std=25, pos_or_ori=0)

    #trajectorySim(shape="circle", pointsNum=50, sensor_std=6, state0=state0, plotBool=True, printBool=False)

    measureDataPredictor(sensor_std=3, state0=state0, plotType=(1, 2), plotBool=False, printBool=True, state=state)

