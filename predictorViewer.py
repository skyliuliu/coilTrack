# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/19 16:00
desc: 定位结果的显示工具
'''
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from filterpy.stats import plot_covariance


def q2R(q):
    '''
    从四元数求旋转矩阵
    :param q: 四元数
    :return: R 旋转矩阵
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3,     2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3,     1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2,     2 * q2 * q3 + 2 * q0 * q1,     1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])
    return R

def q2Euler(q):
    '''
    从四元数求欧拉角
    :param q: 四元数
    :return: 【np.array】 [pitch, roll, yaw]
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    pitch = math.atan2(2 * q0 * q1 + 2 * q2 * q3, 1 - 2 * q1 * q1 - 2 * q2 * q2)
    roll = math.asin(2 * q0 * q2 - 2 * q3 * q1)
    yaw = math.atan2(2 * q0 * q3 + 2 * q1 * q2, 1 - 2 * q2 * q2 - 2 * q3 * q3)
    return np.array([pitch, roll, yaw]) * 57.3

def plotLM(residual_memory, us):
    '''
    描绘LM算法的残差和u值（LM算法的参数）曲线
    :param residual_memory: 【list】残差列表
    :param us: 【list】u值列表
    :return:
    '''
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


# plt.axis('auto')   # 坐标轴自动缩放

def plotP(predictor, state, index, plotType):
    '''
    描绘UKF算法中误差协方差yz分量的变化过程
    :param state0: 【np.array】预测状态 （7，）
    :param state: 【np.array】真实状态 （7，）
    :param index: 【int】算法的迭代步数
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :return:
    '''
    x, y = plotType
    state_copy = state.copy()  # 浅拷贝真实值，因为后面会修改state
    xtruth = state_copy[:3]  # 获取坐标真实值
    mtruth = q2R(state_copy[3: 7])[:, -1]  # 获取姿态真实值，并转换为z方向的矢量

    pos, q = predictor.ukf.x[:3].copy(), predictor.ukf.x[3: 7]  # 获取预测值，浅拷贝坐标值
    em = q2R(q)[:, -1]
    if plotType == (0, 1):
        plt.ylim(-0.2, 0.4)
        plt.axis('equal')  # 坐标轴按照等比例绘图
    elif plotType == (1, 2):
        xtruth[1] += index * 0.1
        pos[1] += index * 0.1
    else:
        raise Exception("invalid plotType")

    P = predictor.ukf.P[x: y+1, x: y+1]  # 坐标的误差协方差
    plot_covariance(mean=pos[x: y+1], cov=P, fc='g', alpha=0.3, title='胶囊定位过程仿真')
    plt.text(pos[x], pos[y], int(index), fontsize=9)
    plt.plot(xtruth[x], xtruth[y], 'ro')  # 画出真实值
    plt.text(xtruth[x], xtruth[y], int(index), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos[x] + em[x] * scale, pos[y] + em[y] * scale), xytext=(pos[x], pos[y]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[x] + mtruth[x] * scale, xtruth[y] + mtruth[y] * scale),
                 xytext=(xtruth[x], xtruth[y]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))

    # 添加坐标轴标识
    plt.xlabel('{}/m'.format('xyz'[x]))
    plt.ylabel('{}/m'.format('xyz'[y]))
    # 添加网格线
    plt.gca().grid(b=True)
    # 增加固定时间间隔
    plt.pause(0.05)

def plotPos(state0, state, index, plotType):
    '''
    描绘预测位置的变化过程
    :param state0: 【np.array】预测状态 （7，）
    :param state: 【np.array】真实状态 （7，）
    :param index: 【int】算法的迭代步数
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :return:
    '''
    x, y = plotType
    state_copy = state.copy()  # 浅拷贝真实值，因为后面会修改state
    xtruth = state_copy[:3]  # 获取坐标真实值
    mtruth = q2R(state_copy[3: 7])[:, -1]  # 获取姿态真实值，并转换为z方向的矢量

    pos, q = state0[:3].copy(), state0[3:]    # 获取预测值，浅拷贝坐标值
    em = q2R(q)[:, -1]
    if plotType == (0, 1):
        # 添加坐标轴标识
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.axis('equal')    # 坐标轴按照等比例绘图
        plt.ylim(-0.2, 0.5)
        # plt.gca().set_aspect('equal', adjustable='box')
    elif plotType == (1, 2):
        xtruth[1] += index
        pos[1] += index
        # 添加坐标轴标识
        plt.xlabel('y/m')
        plt.ylabel('z/m')
    else:
        raise Exception("invalid plotType")

    plt.plot(pos[x], pos[y], 'b+')  # 仅描点
    plt.text(pos[x], pos[y], int(index), fontsize=9)
    plt.plot(xtruth[x], xtruth[y], 'ro')  # 画出真实值
    plt.text(xtruth[x], xtruth[y], int(index), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos[x] + em[x] * scale, pos[y] + em[y] * scale), xytext=(pos[x], pos[y]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[x] + mtruth[x] * scale, xtruth[y] + mtruth[y] * scale),
                 xytext=(xtruth[x], xtruth[y]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))

    plt.gca().grid(b=True)
    plt.pause(0.05)

def plotErr(x, y, z, contourBar, titleName):
    '''
    描绘误差分布的等高线图
    :param x: 【np.array】误差分布的x变量 (n, n)
    :param y: 【np.array】误差分布的y变量 (n, n)
    :param z: 【np.array】误差分布的结果 (n, n)
    :param contourBar: 【np.array】等高线的刻度条
    :param titleName: 【string】图的标题名称
    :return:
    '''
    plt.title(titleName)
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.tick_params(labelsize=10)
    plt_contourf = plt.contourf(x, y, z, contourBar, cmap='jet', extend='both')    # 填充等高线内区域
    cmap = copy.copy(plt_contourf.get_cmap())
    cmap.set_over('red')     # 超过contourBar的上限就填充为red
    cmap.set_under('blue')     # 低于contourBar的下限就填充为blue
    plt_contourf.changed()

    cntr = plt.contour(x, y, z, contourBar, colors='black', linewidths=0.5)    # 描绘等高线轮廓
    plt.clabel(cntr, inline_spacing=1, fmt='%.2f', fontsize=8, colors='black')     # 标识等高线的数值
    plt.show()

def plotTrajectory(stateLine, stateMP):
    '''
    描绘轨迹预估图
    :param stateLine: 【np.array】真实状态的轨迹
    :param stateMP: 【np.array】预估的状态
    :return:
    '''
    plt.axis('equal')  # 坐标轴按照等比例绘图
    plt.gca().grid(b=True)
    plt.plot(stateLine[:, 0], stateLine[:, 1],'r')
    plt.plot(stateMP[:, 0], stateMP[:, 1], 'b--')
    plt.show()