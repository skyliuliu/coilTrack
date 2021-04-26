import datetime
import math

import numpy as np
import matplotlib.pyplot as plt

from calculatorUKF import inducedVolatage
from predictorViewer import q2R, plotPos, plotLM, plotErr


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

# def inducedVolatage(n1=200, nr1=10, n2=20, nr2=4, r1=5, d1=0.2, r2=2.5, d2=0.02, i=5, freq=20000, d=(0, 0, 0.2),
#                     em1=(0, 0, 1), em2=(0, 0, 1)):
#     """
#     计算发射线圈在接收线圈中产生的感应电动势
#     **************************
#     *假设：                   *
#     *1、线圈均可等效为磁偶极矩  *
#     *2、线圈之间静止           *
#     **************************
#     :param n1: 发射线圈匝数 [1]
#     :param nr1: 发射线圈层数 [1]
#     :param n2: 接收线圈匝数 [1]
#     :param nr2: 接收线圈层数 [1]
#     :param r1: 发射线圈内半径 [mm]
#     :param d1: 发射线圈线径 [mm]
#     :param r2: 接收线圈内半径 [mm]
#     :param d2: 接收线圈线径 [mm]
#     :param i: 激励电流的幅值 [A]
#     :param freq: 激励信号的频率 [Hz]
#     :param d: 初级线圈中心到次级线圈中心的位置矢量 [m]
#     :param em1: 发射线圈的朝向 [1]
#     :param em2: 接收线圈的朝向 [1]
#     :return E: 感应电压 [1e-6V]
#     """
#     dNorm = np.linalg.norm(d)
#     er = d / dNorm
#
#     em1 /= np.linalg.norm(em1)
#     em2 /= np.linalg.norm(em2)
#
#     # 精确计算线圈的面积，第i层线圈的面积为pi * (r + d * i) **2
#     S1 = n1 // nr1 * math.pi * sum([(r1 + d1 * j) ** 2 for j in range(nr1)]) / 1000000
#     S2 = n2 // nr2 * math.pi * sum([(r2 + d2 * k) ** 2 for k in range(nr2)]) / 1000000
#
#     B = np.divide(pow(10, -7) * i * S1 * (3 * np.dot(er, em1) * er - em1), dNorm ** 3)
#
#     E = 2 * math.pi * pow(10, -7) * freq * i * S1 * S2 / dNorm ** 3 * (
#                 3 * np.dot(er, em1) * np.dot(er, em2) - np.dot(em1, em2))
#     return E * 1000000  # 单位1e-6V

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
    :return: None
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
        poss.append(state[:3])
        ems.append(state[3:])
        i += 1
        while True:
            Hessian_LM = A + u * np.eye(n)  # calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
            if np.linalg.norm(step) <= eps_step:
                stateOut(state, state2, t0, i, mse, 'threshold_step', printBool)
                return
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
                    return
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

def generate_data(num_data, state):
    """
    生成模拟数据
    :param num_data: 数据维度 [int]
    :return: 模拟的B值, (27, )
    """
    Bmid = h(state)  # 模拟数据的中间值
    std = 5
    Bsim = np.zeros(num_data)

    for j in range(num_data):
        Bsim[j] = np.random.normal(Bmid[j], std, 1)
    return Bsim

def plotLM0(residual_memory, us):
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # plt.plot(residual_memory)
    for ax in [ax1, ax2]:
        ax.set_xlabel("iter")
    ax1.set_ylabel("residual")
    ax1.semilogy(residual_memory)
    ax2.set_xlabel("iter")
    ax2.set_ylabel("u")
    ax2.semilogy(us)
    plt.show()

def plotP0(state0, state, index):
    pos, em = state0[:3], state0[3:]
    emNorm = np.linalg.norm(em)
    em /= emNorm
    xtruth = state.copy()[:3]
    xtruth[1] += index  # 获取坐标真实值
    mtruth = state.copy()[3:]  # 获取姿态真实值
    pos2 = np.zeros(2)
    pos2[0], pos2[1] = pos[1] + index, pos[2]  # 预测的坐标值

    # plt.axis('equal')
    # plt.ylim(0.2, 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(pos2[0], pos2[1], 'b+')
    plt.text(pos2[0], pos2[1], int(index * 10), fontsize=9)
    plt.plot(xtruth[1], xtruth[2], 'ro')  # 画出真实值
    plt.text(xtruth[1], xtruth[2], int(index * 10), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos2[0] + em[1] * scale, pos2[1] + em[2] * scale), xytext=(pos2[0], pos2[1]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[1] + mtruth[1] * scale, xtruth[2] + mtruth[2] * scale),
                 xytext=(xtruth[1], xtruth[2]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))
    # 添加坐标轴标识
    plt.xlabel('iter/1')
    plt.ylabel('pos/m')
    plt.gca().grid(b=True)
    plt.pause(0.02)


def sim(states, state0, sensor_std, plotType, plotBool, printBool, maxIter=100):
    '''
    使用模拟的观测值验证算法的准确性
    :param states: 模拟的真实状态
    :param state0: 模拟的初始值
    :param sensor_std: sensor的噪声标准差[mG]
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    '''
    m, n = coilcols * coilcols, 7
    for i in range(1):
        # run
        output_data = generate_data(m, states[i])
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

def simErrDistributed(contourBar, sensor_std=10, pos_or_ori=1):
    '''
    模拟误差分布
    :param contourBar: 【np.array】等高线的刻度条
    :param sensor_std: 【float】sensor的噪声标准差[mG]
    :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
    :return:
    '''
    n = 20
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, n), np.linspace(-0.2, 0.2, n))
    state0 = np.array([0, 0, 0.3, 1, 0, 0, 0, 0, 0])
    states = [np.array([0, 0, 0.3, 0.5 * math.sqrt(3), 0.5, 0, 0])]
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
    # state0 = np.array([0, 0, 0.3, 1, 0, 0, 0, 0, 0])  # 初始值
    # states = [np.array([0.1, 0.1, 0.3, 0.5 * math.sqrt(3), 0.5, 0, 0])]  # 真实值
    # err = sim(states, state0, sensor_std=25, plotBool=False, plotType=(1, 2), printBool=True)

    simErrDistributed(contourBar=9, sensor_std=100, pos_or_ori=0)