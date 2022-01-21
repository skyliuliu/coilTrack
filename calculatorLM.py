import datetime
import math
from queue import Queue
from multiprocessing.dummy import Process

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from coilArray import CoilArray
from calculatorUKF import generateEsim, trajectoryLine
from predictorViewer import q2R, plotPos, plotLM, plotErr, plotTrajectory
from readData import readRecData, findPeakValley

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tao = 1e-3
delta = 0.0001
eps_stop = 1e-9
eps_step = 1e-6
eps_residual = 1e-3
residual_memory = []
us = []
poss = []
ems = []

def get_init_u(A, tao):
    """
    确定LM算法中u的初始值
    :param A: J.T * J (m, m)
    :param tao: 常量
    :return: u [int]
    """
    m = np.shape(A)[0]
    Aii = []
    for i in range(0, m):
        Aii.append(A[i, i])
    u = tao * max(Aii)
    return u


class Tracker:
    currents = [2.15, 2.18, 2.26, 2.33, 2.27, 2.25, 2.24, 2.32, 2.22, 2.34, 2.31, 2.27, 2.3, 2.3, 2.38, 2.28]
    coils = CoilArray(np.array(currents) + 0.54)

    maxIter = 50        # 最大迭代次数
    printBool = True    # 是否打印结果

    def __init__(self, state):
        self.state = state[:7]
        self.n = len(self.state)
        self.m = self.coils.coilNum

    def derive(self, param_index):
        """
        指定状态量的偏导数
        :param state: 预估的状态量 (n, )
        :param param_index: 第几个状态量
        :return: 偏导数 (m, )
        """
        state1 = self.state.copy()
        state2 = self.state.copy()
        if param_index < 3:
            delta = 0.0003
        else:
            delta = 0.001
        state1[param_index] += delta
        state2[param_index] -= delta
        data_est_output1 = self.coils.h(state1)
        data_est_output2 = self.coils.h(state2)
        return 0.5 * (data_est_output1 - data_est_output2) / delta

    def jacobian(self):
        """
        计算预估状态的雅可比矩阵
        :return: J (m, n)
        """
        J = np.zeros((self.m, self.n))
        for pi in range(0, self.n):
            J[:, pi] = self.derive(pi)
        return J

    def residual(self, state, output_data):
        """
        计算残差
        :param state: 预估的状态量 (n, )
        :param output_data: 观测量 (m, )
        :return: residual (m, )
        """
        data_est_output = self.coils.h(state)
        residual = output_data - data_est_output
        return residual

    def LM(self, state2, output_data):
        """
        Levenberg–Marquardt优化算法的主体
        :param state2: 预估的状态量 (n, ) + [moment, costTime]
        :param output_data: 观测量 (m, )
        :param maxIter: 最大迭代次数
        :return: 【np.array】优化后的状态 (7, )
        """
        output_data = np.array(output_data)
        self.state = np.array(state2)[:7]
        t0 = datetime.datetime.now()
        res = self.residual(self.state, output_data)
        J = self.jacobian()
        A = J.T.dot(J)
        g = J.T.dot(res)
        u = get_init_u(A, tao)  # set the init u
        # u = 100
        v = 2
        rou = 0
        mse = 0

        for i in range(self.maxIter):
            # 如果预测的z坐标为负，则反向，并对方向矢量取y轴对称（经验总结，暂未确定原因）
            if self.state[2] < 0:
                self.state[2] = -self.state[2]
                self.state[-2] = -self.state[-2]
            poss.append(self.state[:3])
            ems.append(self.state[3:])
            i += 1
            while True:
                Hessian_LM = A + u * np.eye(self.n)  # calculating Hessian matrix in LM
                step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
                if np.linalg.norm(step) <= eps_step:
                    self.stateOut(state2, t0, i, mse, 'threshold_step')
                    return self.state
                newState = self.state + step
                newRes = self.residual(newState, output_data)
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
                    us.append(u)
                    residual_memory.append(mse)
                    if stop:
                        self.stateOut(state2, t0, i, mse, 'threshold_stop or threshold_residual')
                        return self.state
                    else:
                        break
                else:
                    u *= v
                    v *= 2
                    us.append(u)
                    residual_memory.append(mse)
            if i == self.maxIter:
                print('maxIter_step')
                self.stateOut(state2, t0, i, mse, ' ')
                return self.state

    def stateOut(self, state2, t0, i, mse, printStr):
        '''
        输出算法的中间结果
        :param state:【np.array】 位置和姿态:x, y, z, q0, q1, q2, q3 (7,)
        :param state2: 【np.array】位置、姿态、磁矩、单步耗时和迭代步数 (10,)
        :param t0: 【float】 时间戳
        :param i: 【int】迭代步数
        :param mse: 【float】残差
        :return:
        '''
        if not self.printBool:
            return
        print(printStr)
        timeCost = (datetime.datetime.now() - t0).total_seconds()
        state2[:] = np.concatenate((self.state, np.array([timeCost, i])))  # 输出的结果
        pos = np.round(self.state[:3], 3)
        em = np.round(q2R(self.state[3: 7])[:, -1], 3)
        # emNorm = np.linalg.norm(state[3:6])
        # em /= emNorm
        # em = np.round(em, 3)
        print('i={}, pos={}m, em={}, timeCost={:.3f}s, mse={:.8e}'.format(i, pos, em, timeCost, mse))

    def compErro(self, state, states):
        '''
        计算预测结果与真实结果之间的相对误差
        :param state: 预测值
        :param states: 真实值
        :return:
        '''
        posTruth, emTruth = states[0][:3], q2R(states[0][3: 7])[:, -1]
        err_pos = np.linalg.norm(state[:3] - posTruth) / np.linalg.norm(posTruth)
        err_em = np.linalg.norm(q2R(state[3: 7]) - emTruth)  # 方向矢量本身是归一化的
        print('pos={}: err_pos={:.0%}, err_em={:.0%}'.format(np.round(posTruth, 3), err_pos, err_em))

        return (err_pos, err_em)

    def generate_data(self, state, std):
        """
        生成模拟数据
        :param num_data: 【int】数据维度
        :param state: 【np.array】真实状态值 (7, )
        :param std: 【float】传感器的噪声标准差
        :return: 模拟值, (num_data, )
        """
        Emid = self.coils.h(state)  # 模拟数据的中间值
        Esim = np.zeros(self.m)

        for j in range(self.m):
            Esim[j] = np.random.normal(Emid[j], std, 1)
        return Esim

    def sim(self, states, state0, sensor_std, plotType, plotBool):
        '''
        使用模拟的观测值验证算法的准确性
        :param states: 真实状态
        :param state0: 初始值
        :param sensor_std: sensor的噪声标准差[mG]
        :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
        :param plotBool: 【bool】是否绘图
        :param printBool: 【bool】是否打印输出
        :param maxIter: 【int】最大迭代次数
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        '''
        for i in range(1):
            # run
            output_data = self.generate_data(states[0][:7], sensor_std)
            self.LM(state0, output_data)

            if plotBool:
                # plot pos and em
                # 最大化窗口
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                iters = len(poss)
                for j in range(iters):
                    state00 = np.concatenate((poss[j], ems[j]))
                    plt.ion()
                    plotPos(state00, states[i], j, plotType)
                    if j == iters - 1:
                        plt.ioff()
                        plt.show()
                # plotLM(residual_memory, us)
                residual_memory.clear()
                us.clear()
                poss.clear()
                ems.clear()

            err_pos, err_em = self.compErro(self.state, states)
            return (err_pos, err_em)

    def run(self, state0):
        '''
        启动实时定位
        :param state0:
        :return:
        '''
        qADC, qGyro, qAcc = Queue(), Queue(), Queue()
        qVpp = []

        # 读取串口数据
        procReadRec = Process(target=readRecData, args=(qADC, qGyro, qAcc))
        procReadRec.daemon = True
        procReadRec.start()

        while True:
            if not qADC.empty():
                adcV = qADC.get()
                vpp = findPeakValley(adcV, 0, 4e-6)
                if vpp:
                    qVpp.append(vpp * 1e6)
            if len(qVpp) == 16:
                self.LM(state0, qVpp)
                qVpp.clear()


    def measureDataPredictor(self, state0, states, plotType, plotBool):
        '''
        使用实测结果估计位姿
        :param states: 真实状态
        :param state0: 初始值
        :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
        :param plotBool: 【bool】是否绘图
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        '''
        # measureData = np.array([19, 37, 30,	12,	53,	105, 82, 29, 61, 129, 103, 35, 32, 62, 48, 19])
        # read measureData.csv
        measureData = np.zeros(self.m)
        with open('measureData.csv' , 'r', encoding='utf-8') as f:
            readData = f.readlines()
        for i in range(self.m):
            measureData[i] = eval(readData[i])
    
        self.LM(state0, measureData)
    
        if plotBool:
            # plot pos and em
            # 最大化窗口
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            iters = len(poss)
            for j in range(iters):
                state00 = np.concatenate((poss[j], ems[j]))
                plt.ion()
                plotPos(state00, states[0], j, plotType)
                if j == iters - 1:
                    plt.ioff()
                    plt.show()

    def funScipy(self, state, coilIndex, Emea):
        d = state[:3] - self.coils.coilArray[coilIndex, :]
        em2 = q2R(state[3: 7])[:, -1]
        #print('pos={}, em={}'.format(state[:3], em2))
    
        Eest = np.zeros(self.m)
        for i in range(self.m):
            Eest[i] = self.coils.inducedVolatage(d=d[i], em2=em2, ii=self.currents[i])
    
        return Eest - Emea

    def simScipy(self, states, state0, sensor_std):
        '''
        使用模拟的观测值验证scipy自带的优化算法
        :param states: 模拟的真实状态
        :param state0: 模拟的初始值
        :param sensor_std: sensor的噪声标准差[mG]
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        '''
        output_data = self.generate_data(states[0], sensor_std)
        result = least_squares(self.funScipy, state0, verbose=0, args=(np.arange(self.m), output_data), xtol=1e-6, jac='3-point',)
        #print(result)
        stateResult = result.x

        err_pos, err_em = self.compErro(stateResult, states)

    def measureDataPredictorScipy(self, state0, states):
        '''
        使用实测结果估计位姿
        :param states: 真实状态
        :param state0: 初始值
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        '''
        # measureData = np.array([19, 37, 30,	12,	53,	105, 82, 29, 61, 129, 103, 35, 32, 62, 48, 19])
        # read measureData.csv
        measureData = np.zeros(self.m)
        with open('measureData.csv' , 'r', encoding='utf-8') as f:
            readData = f.readlines()
        for i in range(self.m):
            measureData[i] = eval(readData[i])

        result = least_squares(self.funScipy, state0, verbose=0, args=(np.arange(self.m), measureData), xtol=1e-6, jac='3-point',)
        stateResult = result.x
        self.compErro(stateResult, states)

    def trajectorySim(self, shape, pointsNum, state0, sensor_std, plotBool):
        line = trajectoryLine(shape, pointsNum)
        q = [0, 0, 1, 1]
        stateLine = np.array([line[i] + q for i in range(pointsNum)])
        state = line[0] + q

        resultList = []  # 储存每一轮优化算法的最终结果

        # 先对初始状态进行预估
        E0sim = self.generate_data(sensor_std)
        result = self.LM(state0, E0sim)
        resultList.append(result)

        # 对轨迹线上的其它点进行预估
        for i in range(1, pointsNum):
            print('--------point:{}---------'.format(i))
            state = line[i] + q
            Esim = self.generate_data(state, sensor_std)
            state2 = np.concatenate((result, [0, 0]))
            result = self.LM(state2, Esim)
            resultList.append(result)

        if plotBool:
            stateMP = np.asarray(resultList)
            plotTrajectory(stateLine, stateMP, sensor_std)


    def simErrDistributed(self, contourBar, sensor_std=10, pos_or_ori=1):
        '''
        模拟误差分布
        :param contourBar: 【np.array】等高线的刻度条
        :param sensor_std: 【float】sensor的噪声标准差[μV]
        :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
        :return:
        '''
        n = 20
        x, y = np.meshgrid(np.linspace(-0.2, 0.2, n), np.linspace(-0.2, 0.2, n))
        state0 = np.array([0, 0, 0.3, 1, 0, 0, 0, 0, 0])
        states = [np.array([0, 0, 0.3, 0, 1, 0, 0])]
        z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # state0[0] = x[i, j]
                # state0[1] = y[i, j]
                states[0][0] = x[i, j]
                states[0][1] = y[i, j]
                z[i, j] = self.sim(states, state0, sensor_std, plotBool=False, plotType=(0, 1))[pos_or_ori]

        plotErr(x, y, z, contourBar, titleName='sensor_std={}'.format(sensor_std))


if __name__ == '__main__':
    state0 = np.array([0, 0, 0.2, 0, 0, 0, 1, 0, 0])  # 初始值
    states = [np.array([0, 0, 0.215 - 0.0075, 1, 0, 0, 0])]  # 真实值

    tracker = Tracker(state0)

    #err = tracker.sim(states, state0, sensor_std=10, plotBool=False, plotType=(1, 2))
    # print('---------------------------------------------------\n')
    # state0S = np.array([0, 0, 0.3, 0, 0, 0, 1])   # 初始值
    # tracker.simScipy(states, state0S, sensor_std=10)

    # simErrDistributed(contourBar=np.linspace(0, 0.5, 9), sensor_std=25, pos_or_ori=0)
    # trajectorySim(shape="straight", pointsNum=50, state0=state0, sensor_std=6, plotBool=True, printBool=True)

    # tracker.measureDataPredictor(state0, states, plotBool=False, plotType=(1, 2))

    # tracker.measureDataPredictorScipy(state0, states)