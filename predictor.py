# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2022/3/3 19:50
desc: 实现定位算法的类，依据状态量的维度有不同的形式
'''
import time
from queue import Queue
from multiprocessing.dummy import Process

import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from calculatorUKF import trajectoryLine
from coilArray import CoilArray
from predictorViewer import q2R, plotErr, plotTrajectory, track3D
from Lie import *
from readData import readRecData

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(suppress=True)


class Predictor:
    printBool = False

    def __init__(self, state, currents):
        '''
        初始化定位类
        :param state: 【np.array】/【se3】初始位姿
        '''
        self.state = state
        if isinstance(state, np.ndarray):
            self.n = len(state)
        elif isinstance(state, se3):
            self.n = len(state.w)
        else:
            raise TypeError("状态量的类型输入有误！")

        self.currents = currents
        self.coils = CoilArray(np.array(currents))
        self.m = self.coils.coilNum + 3

        self.t0 = time.time()
        self.totalTime = 0
        self.compTime = 0
        self.iter = 0

        self.T = self.tR(state)

    def tR(self, state):
        '''
        从状态量解析出位移t和旋转矩阵R，组成变换矩阵
        :return: 【np.array】变换矩阵 (4, 4)
        '''
        T = np.eye(4)
        if self.n == 6:
            t = state.exp().matrix()[:3, 3]
            R = state.exp().matrix()[:3, :3]
        elif self.n == 7:
            t = state[:3]
            R = q2R(state[3: 7])
        else:
            raise ValueError("状态量输入错误")

        T[:3, :3] = R
        T[:3, 3] = t
        return  T

    def h(self, state):
        '''
        纯线圈的观测方程
        :param state: 预估的状态量 (n, )
        :return: E 感应电压 [1e-6V] (m, )
        '''
        T = self.tR(state)
        dArray0 = T[:3, 3] - self.coils.coilArray
        em2 = T[:3, :3][:3, 2]

        E = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            E[i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1, em2=em2, ii=self.currents[i])
        # 正交线圈
        # E = np.zeros(self.m)
        # for i, d in enumerate(dArray0):
        #     E[3*i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1x, em2=em2, ii=self.currents[i])
        #     E[3*i + 1] = self.coils.inducedVolatage(d=d, em1=self.coils.em1y, em2=em2, ii=self.currents[i])
        #     E[3*i + 2] = self.coils.inducedVolatage(d=d, em1=self.coils.em1z, em2=em2, ii=self.currents[i])
        return E

    def hh(self, state):
        """
        线圈+IMU的观测方程
        :param state: 预估的状态量 (n, )
        :return: E+A 感应电压 [1e-6V] + 方向矢量[1] (m, )
        """
        T = self.tR(state)
        dArray0 = T[:3, 3] - self.coils.coilArray
        em2 = T[:3, :3][:3, 2]

        EA = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            EA[i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1, em2=em2, ii=self.currents[i])

        EA[-3] = -1000 * em2[-3]  # x反向
        EA[-2] = -1000 * em2[-2]  # y反向
        EA[-1] = 1000 * em2[-1]   # z正向
        return EA

    def generateData(self, stateX, std):
        '''
        生成模拟数据
        :param stateX: 【np.array】/【se3】真实位姿
        :param std: 【float】传感器输出值的标准差/误差比
        :return:
        '''
        TX = self.tR(stateX)
        midData = self.hh(stateX)
        t = TX[:3, 3]
        ez = TX[2, :3]

        simData = np.zeros(self.m)
        for j in range(self.m):
            #simData[j] = np.random.normal(midData[j], std, 1)
            simData[j] = midData[j] * (1 + (-1) ** j * std)
            # simData[j] = midData[j] + (-1) ** j * std

        if self.printBool:
            print('turth: t={}, ez={}'.format(np.round(t, 3), np.round(ez, 3)))
            print('sensor_mid={}'.format(np.round(midData[:], 3)))
            print('sensor_sim={}'.format(np.round(simData[:], 3)))
        return simData

    def residual(self, state, measureData):
        '''
        计算残差
        :param state: 【np.array】/【se3】预测位姿
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
        delta = 0.001
        if isinstance(self.state, se3):
            state1_w = self.state.w.copy()
            state2_w = self.state.w.copy()
            state1_w[param_index] += delta
            state2_w[param_index] -= delta
            state1 = se3(vector=state1_w)
            state2 = se3(vector=state2_w)
        else:
            state1 = self.state.copy()
            state2 = self.state.copy()
            state1[param_index] += delta
            state2[param_index] -= delta

        data_est_output1 = self.hh(state1)
        data_est_output2 = self.hh(state2)
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

    def LM(self, measureData, maxIter=100):
        """
        Levenberg–Marquardt优化算法的主体
        :param maxIter: 最大迭代次数
        :param measureData: 【np.array】 测量值 (self.m, )
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

                if isinstance(self.state, se3):
                    newState = se3(vector=self.state.vector() + step)   # 先将se3转换成数组相加，再变回李代数，这样才符合LM算法流程
                else:
                    newState = self.state + step
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

        T = self.tR(self.state)
        t = np.round(T[:3, 3], 1)
        em = np.round(T[2, :3], 3)
        print('{}:\ni={}, t={}mm, em={}, compTime={:.3f}s, cost={:.3e}'.format(printStr, i, t, em, self.compTime, mse))

    def compErro(self, state, stateX):
        '''
        计算预测结果与真实结果之间的相对误差
        :param state: 【np.array】/【se3】预测位姿
        :param stateX: 【np.array】/【se3】真实位姿
        :return:
        '''
        T = self.tR(state)
        t = T[:3, 3]
        ez = T[2, :3]
        TX = self.tR(stateX)
        err_pos = np.linalg.norm(T[:3, 3] - TX[:3, 3])   # 位移之差的模
        err_em = np.arccos(np.dot(ez, TX[2, :3]) / np.linalg.norm(ez) / np.linalg.norm(TX[2, :3])) * 57.3   # 方向矢量形成的夹角
        print('\nt={}mm, ez={}: err_t={:.0f}mm, err_em={:.1f}deg'.format(np.round(t, 1), np.round(ez, 3), err_pos, err_em))

        return (err_pos, err_em)

    def sim(self, sensor_std, stateX):
        '''
        :param sensor_std: 【float】sensor的噪声标准差[μV]或者误差百分比
        :param stateX: 【np.array】/【se3】真实位姿
        使用模拟的观测值验证算法的准确性
        :return:
        '''
        measureData = self.generateData(std=sensor_std, stateX=stateX)
        self.LM(measureData)

        err_pos, err_em = self.compErro(self.state, stateX)
        return (err_pos, err_em)

    def measureDataPredictor(self, state0):
        '''
        使用实测结果估计位姿
        :param state0: 【np.array】/【se3】初始位姿
        :return:
        '''
        # measureData = np.array([19, 37, 30,	12,	53,	105, 82, 29, 61, 129, 103, 35, 32, 62, 48, 19])
        # read measureData.csv
        measureData = np.zeros(self.m)
        with open('measureData.csv', 'r', encoding='utf-8') as f:
            readData = f.readlines()
        for i in range(self.m):
            measureData[i] = eval(readData[i])

        self.LM(measureData)

    def funScipy(self, state, coilIndex, measureData):
        '''
        误差Cost计算函数
        :param state: 【np.array】估计位姿
        :param coilIndex: 【int】线圈编号
        :param measureData: 【np.array】测量值 (self.m,)
        :return: 【np.array】估计值-测量值 (self.m,)
        '''
        T = self.tR(state)
        d = T[:3, 3] - self.coils.coilArray[coilIndex, :]
        em2 = T[2, :3]

        Eest = np.zeros(self.m)
        for i in range(self.m):
            Eest[i] = self.coils.inducedVolatage(d=d[i], em1=self.coils.em1, em2=em2, ii=self.currents[i])

        return Eest - measureData

    def simScipy(self, stateX, sensor_std):
        '''
        使用模拟的观测值验证scipy自带的优化算法
        :param stateX: 【np.array】模拟的真实状态
        :param sensor_std: sensor的噪声标准差[mG]
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        '''
        output_data = self.generateData(stateX, sensor_std)
        result = least_squares(self.funScipy, self.state, verbose=0, args=(np.arange(self.m), output_data), xtol=1e-6,
                               jac='3-point', )
        stateResult = result.x

        err_t, err_em = self.compErro(stateResult, stateX)
        return (err_t, err_em)

    def measureDataPredictorScipy(self, stateX):
        '''
        使用实测结果估计位姿
        :param stateX: np.array】真实状态
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        '''
        # measureData = np.array([19, 37, 30,	12,	53,	105, 82, 29, 61, 129, 103, 35, 32, 62, 48, 19])
        # read measureData.csv
        measureData = np.zeros(self.m)
        with open('measureData.csv' , 'r', encoding='utf-8') as f:
            readData = f.readlines()
        for i in range(self.m):
            measureData[i] = eval(readData[i])

        result = least_squares(self.funScipy, self.state, verbose=0, args=(np.arange(self.m), measureData), xtol=1e-6, jac='3-point',)
        stateResult = result.x
        err_t, err_em = self.compErro(stateResult, stateX)

    def trajectorySim(self, shape, pointsNum, sensor_std, plotBool):
        '''
        模拟某条轨迹下的定位效果
        :param shape: 【string】形状
        :param pointsNum: 【int】线上的点数
        :param sensor_std: 【float】传感器噪声，此处指感应电压的采样噪声[μV]
        :param plotBool: 【bool】是否绘图
        :param maxIter: 【int】最大迭代次数
        :return:
        实现流程
        1、定义轨迹
        2、提取轨迹上的点生成模拟数据
        3、提取预估的状态，并绘制预估轨迹
        '''
        line = trajectoryLine(shape, pointsNum)
        q = [1, 0, 0, 0]
        stateLine = np.array([line[i] + q for i in range(pointsNum)])
        state = line[0] + q

        resultList = []  # 储存每一轮优化算法的最终结果

        # 先对初始状态进行预估
        E0sim = self.generateData(stateX=state, std=sensor_std)
        self.LM(E0sim)
        resultList.append(self.state.copy())

        # 对轨迹线上的其它点进行预估
        for i in range(1, pointsNum):
            print('--------point:{}---------'.format(i))
            state = line[i] + q
            Esim = self.generateData(stateX=state, std=sensor_std)
            self.LM(Esim)
            resultList.append(self.state.copy())

        if plotBool:
            stateMP = np.asarray(resultList)
            plotTrajectory(stateLine, stateMP, sensor_std)

    def simErrDistributed(self, contourBar, sensor_std=0.01, pos_or_ori=0):
        '''
        模拟误差分布
        :param contourBar: 【np.array】等高线的刻度条
        :param sensor_std: 【float】sensor的噪声标准差[μV]
        :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
        :return:
        '''
        n = 10
        x, y = np.meshgrid(np.linspace(-200, 200, n), np.linspace(-200, 200, n))
        state0 = np.array([0, 0, 200, 1, 0, 0, 0], dtype=float)
        stateX = np.array([0, 0, 200, 1, 1, 0, 0], dtype=float)
        z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # state0[0] = x[i, j]
                # state0[1] = y[i, j]
                stateX[0] = x[i, j]
                stateX[1] = y[i, j]
                z[i, j] = self.simScipy(sensor_std=sensor_std, stateX=stateX)[pos_or_ori]
                self.state = state0

        plotErr(x, y, z, contourBar, titleName='sensor_std={}'.format(sensor_std))

def sim():
    '''
    仿真
    :return:
    '''
    state0 = np.array([0, 0, 200, 1, 0, 0, 0], dtype=float)  # 初始值
    states = np.array([3.6, -147.7, 162.7, 0.81008725, 0.58584571, 0.00421135, 0.02292847], dtype=float)  # 真实值
    # states = np.array([-50, 0, 200, 1, 0, 0, 0], dtype=float)

    state = se3(vector=np.array([0, 0, 0, 0, 0, 300]))
    stateX = se3(vector=np.array([1.252, 0.009, 0.049, -0.066, -26.071, 233.36]))

    pred = Predictor(state=state0, currents=[2] * 16)

    pred.sim(sensor_std=0, stateX=states)

    #pred.simScipy(stateX=states, sensor_std=0.02)

    # pred.trajectorySim(shape="circle", pointsNum=20, sensor_std=0.02, plotBool=True)

    # pred.simErrDistributed(contourBar=np.linspace(0, 50, 9), sensor_std=0.02, pos_or_ori=0)

def run():
    '''
    启动实时定位
    :return:
    '''
    qADC, qGyro, qAcc = Queue(), Queue(), Queue()
    state = np.array([0, 0, 200, 1, 0, 0, 0], dtype=float)
    state_se3 = se3(vector=np.array([0, 0, 0, 0, 0, 200]))

    # 读取接收端数据
    procReadRec = Process(target=readRecData, args=(qADC, qGyro, qAcc))
    procReadRec.daemon = True
    procReadRec.start()
    time.sleep(0.5)

    # 读取发射端的电流，然后创建定位器对象
    currents = [2.22, 2.2, 2.31, 2.37, 2.32, 2.26, 2.26, 2.37, 2.24, 2.37, 2.36, 2.32, 2.34, 2.42, 2.41, 2.3]
    # runsend(open=True)
    pred = Predictor(state, currents)

    # 描绘3D轨迹
    track3D(state, qList=[qADC, qGyro, qAcc], tracker=pred)

if __name__ == '__main__':
    run()