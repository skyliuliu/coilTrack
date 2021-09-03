import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from calculatorUKF import generateEsim, trajectoryLine, inducedVolatage, solenoid
from predictorViewer import q2R, plotPos, plotLM, plotErr, plotTrajectory

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

distance = 0.1  # 初级线圈之间的距离[m]
coilrows = 4
coilcols = 4
CAlength = distance * (coilrows - 1)
CAwidth = distance * (coilcols - 1)
coilArray = np.zeros((coilrows * coilcols, 3))
for row in range(coilrows):
    for col in range(coilcols):
        coilArray[row * coilrows + col] = np.array(
            [-0.5 * CAlength + distance * col, 0.5 * CAwidth - distance * row, 0])


def derive(state, param_index):
    """
    指定状态量的偏导数
    :param state: 预估的状态量 (n, )
    :param param_index: 第几个状态量
    :return: 偏导数 (m, )
    """
    state1 = state.copy()
    state2 = state.copy()
    if param_index < 3:
        delta = 0.0003
    else:
        delta = 0.001
    state1[param_index] += delta
    state2[param_index] -= delta
    data_est_output1 = h(state1)
    data_est_output2 = h(state2)
    return 0.5 * (data_est_output1 - data_est_output2) / delta


def h(state):
    """
    观测方程
    :param state: 预估的状态量 (n, )
    :param m: 观测量的个数 [int]
    :return: E 感应电压 [1e-6V] (m, )
    """
    dArray0 = state[:3] - coilArray
    em2 = q2R(state[3: 7])[:, -1]
    # emNorm = np.linalg.norm(em2)
    # em2 /= emNorm
    E = np.zeros(coilcols * coilrows)
    for i, d in enumerate(dArray0):
        # E[i * 3] = inducedVolatage(d=d, em1=(1, 0, 0), em2=em2)
        # E[i * 3 + 1] = inducedVolatage(d=d, em1=(0, 1, 0), em2=em2)
        # E[i * 3 + 2] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
        E[i] = inducedVolatage(d=d, em1=(0, 0, 1), em2=em2)
    return E


def jacobian(state, m):
    """
    计算预估状态的雅可比矩阵
    :param state: 预估的状态量 (n, )
    :param m: 观测量的个数 [int]
    :return: J (m, n)
    """
    n = state.shape[0]
    J = np.zeros((m, n))
    for pi in range(0, n):
        J[:, pi] = derive(state, pi)
    return J


def residual(state, output_data):
    """
    计算残差
    :param state: 预估的状态量 (n, )
    :param output_data: 观测量 (m, )
    :return: residual (m, )
    """
    data_est_output = h(state)
    residual = output_data - data_est_output
    return residual


def get_init_u(A, tao):
    """
    确定u的初始值
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


def LM(state2, output_data, maxIter, printBool):
    """
    Levenberg–Marquardt优化算法的主体
    :param state2: 预估的状态量 (n, ) + [moment, costTime]
    :param output_data: 观测量 (m, )
    :param maxIter: 最大迭代次数
    :return: 【np.array】优化后的状态 (7, )
    """
    output_data = np.array(output_data)
    state = np.array(state2)[:7]
    t0 = datetime.datetime.now()
    m = output_data.shape[0]
    n = state.shape[0]
    res = residual(state, output_data)
    J = jacobian(state, m)
    A = J.T.dot(J)
    g = J.T.dot(res)
    u = get_init_u(A, tao)  # set the init u
    # u = 100
    v = 2
    rou = 0
    mse = 0

    for i in range(maxIter):
        # 如果预测的z坐标为负，则反向，并对方向矢量取y轴对称（经验总结，暂未确定原因）
        if state[2] < 0:
            state[2] = -state[2]
            state[-2] = -state[-2]
        poss.append(state[:3])
        ems.append(state[3:])
        i += 1
        while True:
            Hessian_LM = A + u * np.eye(n)  # calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
            if np.linalg.norm(step) <= eps_step:
                stateOut(state, state2, t0, i, mse, 'threshold_step', printBool)
                return state
            newState = state + step
            newRes = residual(newState, output_data)
            mse = np.linalg.norm(res) ** 2
            newMse = np.linalg.norm(newRes) ** 2
            rou = (mse - newMse) / (step.T.dot(u * step + g))
            if rou > 0:
                state = newState
                res = newRes
                J = jacobian(state, m)
                A = J.T.dot(J)
                g = J.T.dot(res)
                u *= max(1 / 3, 1 - (2 * rou - 1) ** 3)
                v = 2
                stop = (np.linalg.norm(g, ord=np.inf) <= eps_stop) or (mse <= eps_residual)
                us.append(u)
                residual_memory.append(mse)
                if stop:
                    stateOut(state, state2, t0, i, mse, 'threshold_stop or threshold_residual', printBool)
                    return state
                else:
                    break
            else:
                u *= v
                v *= 2
                us.append(u)
                residual_memory.append(mse)
        if i == maxIter:
            print('maxIter_step')
            stateOut(state, state2, t0, i, mse, ' ', printBool)
            return state


def stateOut(state, state2, t0, i, mse, printStr, printBool):
    '''
    输出算法的中间结果
    :param state:【np.array】 位置和姿态:x, y, z, q0, q1, q2, q3 (7,)
    :param state2: 【np.array】位置、姿态、磁矩、单步耗时和迭代步数 (10,)
    :param t0: 【float】 时间戳
    :param i: 【int】迭代步数
    :param mse: 【float】残差
    :return:
    '''
    if not printBool:
        return
    print(printStr)
    timeCost = (datetime.datetime.now() - t0).total_seconds()
    state2[:] = np.concatenate((state, np.array([timeCost, i])))  # 输出的结果
    pos = np.round(state[:3], 3)
    em = np.round(q2R(state[3: 7])[:, -1], 3)
    # emNorm = np.linalg.norm(state[3:6])
    # em /= emNorm
    # em = np.round(em, 3)
    print('i={}, pos={}m, em={}, timeCost={:.3f}s, mse={:.8e}'.format(i, pos, em, timeCost, mse))


def generate_data(num_data, state, std):
    """
    生成模拟数据
    :param num_data: 【int】数据维度
    :param state: 【np.array】真实状态值 (7, )
    :param std: 【float】传感器的噪声标准差
    :return: 模拟值, (num_data, )
    """
    Emid = h(state)  # 模拟数据的中间值
    Esim = np.zeros(num_data)

    for j in range(num_data):
        Esim[j] = np.random.normal(Emid[j], std, 1)
    return Esim


def sim(states, state0, sensor_std, plotType, plotBool, printBool, maxIter=100):
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
    m, n = coilrows * coilcols, 7
    for i in range(1):
        # run
        output_data = generate_data(m, states[i], sensor_std)
        LM(state0, output_data, maxIter, printBool)

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

        posTruth, emTruth = states[0][:3], q2R(states[0][3: 7])[:, -1]
        err_pos = np.linalg.norm(poss[-1] - posTruth) / np.linalg.norm(posTruth)
        err_em = np.linalg.norm(q2R(ems[-1])[:, -1] - emTruth)  # 方向矢量本身是归一化的
        print('pos={}: err_pos={:.0%}, err_em={:.0%}'.format(np.round(posTruth, 3), err_pos, err_em))
        residual_memory.clear()
        us.clear()
        poss.clear()
        ems.clear()
        return (err_pos, err_em)

def measureDataPredictor(state0, states, plotType, plotBool, printBool, maxIter=100):
    '''
    使用实测结果估计位姿
    :param states: 真实状态
    :param state0: 初始值
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    '''
    # measureData = np.array([19, 37, 30,	12,	53,	105, 82, 29, 61, 129, 103, 35, 32, 62, 48, 19])
    # read measureData.csv
    measureData = np.zeros(coilrows * coilcols)
    with open('measureData.csv' , 'r', encoding='utf-8') as f:
        readData = f.readlines()
    for i in range(coilrows * coilcols):
        measureData[i] = eval(readData[i])

    LM(state0, measureData, maxIter, printBool)

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

def funScipy(state, coilIndex, Emea):
    d = state[:3] - coilArray[coilIndex, :]
    em2 = q2R(state[3: 7])[:, -1]
    #print('pos={}, em={}'.format(state[:3], em2))

    Eest = np.zeros(coilrows * coilcols)
    for i in range(coilrows * coilcols):
        Eest[i] = inducedVolatage(d=d[i], em2=em2)

    return Eest - Emea


def simScipy(states, state0, sensor_std):
    '''
    使用模拟的观测值验证scipy自带的优化算法
    :param states: 模拟的真实状态
    :param state0: 模拟的初始值
    :param sensor_std: sensor的噪声标准差[mG]
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    '''
    m, n = coilrows * coilcols, 7
    output_data = generate_data(m, states[0], sensor_std)
    result = least_squares(funScipy, state0, verbose=0, args=(np.arange(m), output_data), xtol=1e-6, jac='3-point',)
    #print(result)
    stateResult = result.x

    pos = np.round(stateResult[:3], 3)
    em = np.round(q2R(stateResult[3: 7])[:, -1], 3)
    posTruth, emTruth = states[0][:3], q2R(states[0][3: 7])[:, -1]
    err_pos = np.linalg.norm(pos - posTruth) / np.linalg.norm(posTruth)
    err_em = np.linalg.norm(em - emTruth)  # 方向矢量本身是归一化的
    print('pos={}, em={} : err_pos={:.0%}, err_em={:.0%}'.format(pos, em, err_pos, err_em))

def measureDataPredictorScipy(state0, states, maxIter=100):
    '''
    使用实测结果估计位姿
    :param states: 真实状态
    :param state0: 初始值
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    '''
    # measureData = np.array([19, 37, 30,	12,	53,	105, 82, 29, 61, 129, 103, 35, 32, 62, 48, 19])
    # read measureData.csv
    m = coilrows * coilcols
    measureData = np.zeros(m)
    with open('measureData.csv' , 'r', encoding='utf-8') as f:
        readData = f.readlines()
    for i in range(m):
        measureData[i] = eval(readData[i])

    result = least_squares(funScipy, state0, verbose=0, args=(np.arange(m), measureData), xtol=1e-6, jac='3-point',)

    stateResult = result.x
    pos = np.round(stateResult[:3], 3)
    em = np.round(q2R(stateResult[3: 7])[:, -1], 3)
    posTruth, emTruth = states[0][:3], q2R(states[0][3: 7])[:, -1]
    err_pos = np.linalg.norm(pos - posTruth) / np.linalg.norm(posTruth)
    err_em = np.linalg.norm(em - emTruth)  # 方向矢量本身是归一化的
    print('pos={}, em={} : err_pos={:.0%}, err_em={:.0%}'.format(pos, em, err_pos, err_em))


def trajectorySim(shape, pointsNum, state0, sensor_std, plotBool, printBool, maxIter=100):
    line = trajectoryLine(shape, pointsNum)
    q = [0, 0, 1, 1]
    stateLine = np.array([line[i] + q for i in range(pointsNum)])
    state = line[0] + q

    m = coilrows * coilcols
    resultList = []  # 储存每一轮优化算法的最终结果

    # 先对初始状态进行预估
    E0sim = generate_data(m, state, sensor_std)
    result = LM(state0, E0sim, maxIter, printBool)
    resultList.append(result)

    # 对轨迹线上的其它点进行预估
    for i in range(1, pointsNum):
        print('--------point:{}---------'.format(i))
        state = line[i] + q
        Esim = generate_data(m, state, sensor_std)
        state2 = np.concatenate((result, [0, 0]))
        result = LM(state2, Esim, maxIter, printBool)
        resultList.append(result)

    if plotBool:
        stateMP = np.asarray(resultList)
        plotTrajectory(stateLine, stateMP, sensor_std)


def simErrDistributed(contourBar, sensor_std=10, pos_or_ori=1):
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
            z[i, j] = sim(states, state0, sensor_std, plotBool=False, plotType=(0, 1), printBool=False)[pos_or_ori]

    plotErr(x, y, z, contourBar, titleName='sensor_std={}'.format(sensor_std))


if __name__ == '__main__':
    state0 = np.array([0, 0, 0.3, 0, 0, 0, 1, 0, 0])  # 初始值
    states = [np.array([-0.025, -0.025, 0.25, 1, 0, 0, 0])]  # 真实值
    # err = sim(states, state0, sensor_std=10, plotBool=False, plotType=(1, 2), printBool=True)
    # print('---------------------------------------------------\n')
    # state0S = np.array([0, 0, 0.3, 0, 0, 0, 1])   # 初始值
    # simScipy(states, state0S, sensor_std=10)

    # simErrDistributed(contourBar=np.linspace(0, 0.5, 9), sensor_std=25, pos_or_ori=0)
    # trajectorySim(shape="straight", pointsNum=50, state0=state0, sensor_std=6, plotBool=True, printBool=True)
    # Emind = h(state0)

    # Esim = generate_data(16, state0, 3)
    # print(Esim)

    measureDataPredictor(state0, states, plotBool=False, plotType=(1, 2), printBool=True)

    measureDataPredictorScipy(state0, states, maxIter=100)