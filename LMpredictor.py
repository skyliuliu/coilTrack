# coding=utf-8
# /usr/bin/env python3
"""
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2022/3/3 19:50
desc: 实现定位算法的类，依据状态量的维度有不同的形式
"""
import csv
import time
from multiprocessing.dummy import Process
from queue import Queue

import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from Lie import *
from coilArray import CoilArray
from predictorViewer import q2R, track3D
from readData import readRecData

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(suppress=True, precision=2)


class Predictor:
    printBool = False

    def __init__(self, currents, state0):
        """
        初始化定位类
        :param currents:【float】发射线圈阵列的电流幅值[A]
        :param state0: 【np.array】/【se3】初始位姿
        """
        self.state = state0
        if isinstance(state0, np.ndarray):
            self.n = len(state0)
        elif isinstance(state0, se3):
            self.n = len(state0.w)
        else:
            raise TypeError("状态量的类型输入有误！")

        self.currents = currents
        self.coils = CoilArray(np.array(currents))
        self.m = self.coils.coilNum + 2

        self.t0 = time.time()
        self.totalTime = 0
        self.compTime = 0
        self.iter = 0

        # self.T = self.tR(state)

    def tR(self, state):
        """
        从状态量解析出位移t和旋转矩阵R，组成变换矩阵
        :param state: 【np.array】状态量 (n,)
        :return: 【np.array】变换矩阵 (4, 4)
        """
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
        return T

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
        return np.concatenate((pos, em2))

    def h(self, state):
        """
        观测方程
        :param state: 预估的状态量 (n, )
        :return: E+A 感应电压 [1e-6V] + 方向矢量[1] (m, )
        """
        pos_em2 = self.parseState(state)
        pos, em2 = pos_em2[:3], pos_em2[3:]
        dArray0 = pos - self.coils.coilArray

        EA = np.zeros(self.m)
        for i, d in enumerate(dArray0):
            EA[i] = self.coils.inducedVolatage(d=d, em1=self.coils.em1, em2=em2, ii=self.currents[i])

        if self.m == CoilArray.coilNum:  # 纯线圈
            return EA
        elif self.m == CoilArray.coilNum + 2:  # 基于θ和φ的线圈+IMU
            EA[-1] = 10.24 * np.cos(state[3])
            EA[-2] = 10.24 * np.sin(state[4])
        elif self.m == CoilArray.coilNum + 3:  # 线圈+加速度计
            EA[-3] = 10 * em2[-3]  # x方向
            EA[-2] = 10 * em2[-2]  # y方向
            EA[-1] = 10 * em2[-1]  # z方向
        else:
            raise ValueError("观测量输入错误")
        return EA

    def generateData(self, stateX, std):
        """
        生成模拟数据
        :param stateX: 【np.array】/【se3】真实位姿
        :param std: 【float】传感器输出值的标准差/误差比
        :return:
        """
        pos_em2 = self.parseState(stateX)
        pos, em2 = pos_em2[:3], pos_em2[3:]
        midData = self.h(stateX)

        simData = np.zeros(self.m)
        for j in range(self.m):
            # simData[j] = np.random.normal(midData[j], std, 1)   # 正太分布的噪声模型
            simData[j] = midData[j] * (1 + (-1) ** j * std)  # 百分比误差的噪声模型
            # simData[j] = midData[j] + (-1) ** j * std

        if self.printBool:
            print('turth: t={}, ez={}'.format(np.round(pos, 3), np.round(em2, 3)))
            print('sensor_mid={}'.format(np.round(midData[:], 3)))
            print('sensor_sim={}'.format(np.round(simData[:], 3)))
        return simData

    def residual(self, state, measureData):
        """
        计算残差
        :param state: 【np.array】/【se3】预测位姿
        :param measureData:  【np.array】观测值
        :return: 【np.array】观测值 - 预测值
        """
        eastData = self.h(state)
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

        data_est_output1 = self.h(state1)
        data_est_output2 = self.h(state2)
        return 0.5 * (data_est_output1 - data_est_output2) / delta

    def jacobian(self):
        """
        计算预估状态的雅可比矩阵
        :return: 【np.array (m, n)】雅可比矩阵
        """
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

    def solve(self, measureData, maxIter=100):
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
                    newState = se3(vector=self.state.vector() + step)  # 先将se3转换成数组相加，再变回李代数，这样才符合LM算法流程
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
        """
        输出算法的中间结果
        :param printStr: 【string】终止提示语
        :param t0: 【float】 时间戳
        :param i: 【int】迭代步数
        :param mse: 【float】残差
        :return:
        """
        self.compTime = time.time() - t0
        self.totalTime = time.time() - self.t0
        self.t0 = time.time()
        self.iter = i
        pos_em2 = self.parseState(self.state)
        pos, em2 = pos_em2[:3], pos_em2[3:]

        if not self.printBool:
            return

        print('{}:\ni={}, t={}mm, em2={}, compTime={:.3f}s, cost={:.3e}'.format(printStr, i, pos, em2, self.compTime,
                                                                                mse))

    def compErro(self, state, stateX):
        """
        计算预测结果与真实结果之间的相对误差
        :param state: 【np.array】/【se3】预测位姿
        :param stateX: 【np.array】/【se3】真实位姿
        :return:
        """
        T = self.tR(state)
        t = T[:3, 3]
        ez = T[2, :3]
        TX = self.tR(stateX)
        err_pos = np.linalg.norm(T[:3, 3] - TX[:3, 3])  # 位移之差的模
        err_em = np.arccos(np.dot(ez, TX[2, :3]) / np.linalg.norm(ez) / np.linalg.norm(TX[2, :3])) * 57.3  # 方向矢量形成的夹角
        print('\nt={}mm, ez={}: err_t={:.0f}mm, err_em={:.1f}deg'.format(np.round(t, 1), np.round(ez, 3), err_pos,
                                                                         err_em))

        return err_pos, err_em

    def sim(self, sensor_std, stateX):
        """
        :param sensor_std: 【float】sensor的噪声标准差[μV]或者误差百分比
        :param stateX: 【np.array】/【se3】真实位姿
        使用模拟的观测值验证算法的准确性
        :return:
        """
        measureData = self.generateData(std=sensor_std, stateX=stateX)
        # self.LM(measureData)
        #
        # err_pos, err_em = self.compErro(self.state, stateX)
        return

    def funScipy(self, state, coilIndex, measureData):
        """
        误差Cost计算函数
        :param state: 【np.array】估计位姿
        :param coilIndex: 【int】线圈编号
        :param measureData: 【np.array】测量值 (self.m,)
        :return: 【np.array】估计值-测量值 (self.m,)
        """
        T = self.tR(state)
        d = T[:3, 3] - self.coils.coilArray[coilIndex, :]
        em2 = T[2, :3]

        Eest = np.zeros(self.m)
        for i in range(self.m):
            Eest[i] = self.coils.inducedVolatage(d=d[i], em1=self.coils.em1, em2=em2, ii=self.currents[i])

        return Eest - measureData

    def simScipy(self, stateX, sensor_std):
        """
        使用模拟的观测值验证scipy自带的优化算法
        :param stateX: 【np.array】模拟的真实状态
        :param sensor_std: sensor的噪声标准差[mG]
        :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
        """
        output_data = self.generateData(stateX, sensor_std)
        result = least_squares(self.funScipy, self.state, verbose=0, args=(np.arange(self.m), output_data), xtol=1e-6,
                               jac='3-point', )
        stateResult = result.x

        err_t, err_em = self.compErro(stateResult, stateX)
        return (err_t, err_em)

    def genTranData(self, sensor_std, shape="circle", velocity=5):
        """
        生成动态轨迹的模拟数据
        :param shape: 【string】形状
        :param velocity: 【float】速度[mm/s]
        :param sensor_std: 【float】传感器噪声
        :return:
        """
        dt = 0.05  # 采样间隔时间[s]
        ym = 100  # 位移幅值[mm]

        if shape == "circle":
            pointsNum = int(10 * np.pi * ym / dt / velocity)
            acc = velocity * velocity / ym
            angularVel = 10 * np.pi / dt / pointsNum

            angle = np.linspace(0, 2 * np.pi, pointsNum)
            line_x = np.sin(angle) * ym
            line_y = np.cos(angle) * ym
            line = [(x, y, 300) for (x, y) in zip(line_x, line_y)]
            v_ex = np.cos(angle)
            v_ey = -np.sin(angle)
            v = [(v_ex * velocity, v_ey * velocity, 0) for (v_ex, v_ey) in zip(v_ex, v_ey)]
            a_ex = -np.sin(angle)
            a_ey = -np.cos(angle)
            a = [(a_ex * acc, a_ey * velocity, 0) for (a_ex, a_ey) in zip(a_ex, a_ey)]
            q = [(np.cos(0.5 * ang), 0, 0, np.sin(0.5 * ang)) for ang in angle]
            w = (0, 0, angularVel)

            now = time.time()

            with open('.\data\simData20220615.csv', 'w', newline='') as f:
                fcsv = csv.writer(f)
                fcsv.writerow(
                    ('timeStamp/s', 'positon/mm', 'q', 'velocity(mm/s)', 'coil', 'E/uV', 'accelerator(mm/s^2)',
                     'gyroscope(deg/s)'))
                for i in range(pointsNum):
                    stateX = np.concatenate((line[i], q[i]))
                    Esim = self.generateData(stateX=stateX, std=sensor_std)[i % 16]
                    ai_np = q2R(q[i]).T[:3, 2] * 9800 + np.array(a[i])
                    ai = tuple(_ for _ in ai_np)
                    fcsv.writerow((now + dt * i, line[i], q[i], v[i], i % 16, Esim, ai, w))
        else:
            raise TypeError("shape is not right!!!")


def sim():
    """
    仿真
    :return:
    """
    state0 = np.array([0, 0, 200, 1, 0, 0, 0], dtype=float)  # 初始值
    states = np.array([30, -10, 260, 1, 0, 0, 0], dtype=float)  # 真实值
    # states = np.array([-50, 0, 200, 1, 0, 0, 0], dtype=float)

    state = se3(vector=np.array([0, 0, 0, 0, 0, 300]))
    stateX = se3(vector=np.array([1.252, 0.009, 0.049, -0.066, -26.071, 233.36]))

    pred = Predictor(currents=[2] * 16, state0=state0)

    pred.genTranData(sensor_std=0)


def run():
    '''
    启动实时定位
    :return:
    '''
    qADC, qGyro, qAcc = Queue(), Queue(), Queue()
    state = np.array([0, 0, 200, np.pi / 4, 0], dtype=float)   # x, y, z, θ, φ 
    # state = np.array([0, 0, 200, 0, 0, 0, 1], dtype=float)   # x, y, z, q0, q1, q2, q3
    state_se3 = se3(vector=np.array([0, 0, 0, 0, 0, 200]))

    # 读取接收端数据
    procReadRec = Process(target=readRecData, args=(qADC, qGyro, qAcc))
    procReadRec.daemon = True
    procReadRec.start()
    time.sleep(0.5)

    # 读取发射端的电流，然后创建定位器对象
    currents = [2.22, 2.2, 2.31, 2.37, 2.32, 2.26, 2.26, 2.37, 2.24, 2.37, 2.36, 2.32, 2.34, 2.42, 2.41, 2.3]
    # runsend(open=True)
    pred = Predictor(currents, state)

    # 描绘3D轨迹
    track3D(state, qList=[qADC, qGyro, qAcc], tracker=pred, quaternion=False)


if __name__ == '__main__':
    run()
