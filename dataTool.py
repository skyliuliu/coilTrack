# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/7/4 16:41
desc: 用于实测数据E的处理
'''
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats


def findPeakValley(data, E0):
    '''
    寻峰算法:
    1、对连续三个点a, b, c，若a<b && b>c，且b>E0，则b为峰点；若a>b && b<c，且b<E0，则b为谷点
    2、保存邻近的峰点和谷点，取x=±3个点内的最大或最小值作为该区段的峰点或谷点
    :param data: 【pd】读取的原始数据
    :param E0: 【float】原始数据平均值
    :return:
    '''
    dataSize = len(data)
    #startIndex = data._stat_axis._start
    startIndex = data.index.start
    # 找出满足条件1的峰和谷
    peaks, valleys = [], []
    for i in range(1, dataSize-1):
        d1, d2, d3 = data['E'][startIndex + i-1], data['E'][startIndex + i], data['E'][startIndex + i+1]
        if d1 < d2 and d2 >= d3 and d2 > E0 + 3*noiseStd:
            if not peaks or i - peaks[-1][0] > 6:  # 第一次遇到峰值或距离上一个峰值超过6个数
                peaks.append((i, d2))
            elif peaks[-1][1] < d2:   # 局部区域有更大的峰值
                peaks[-1] = (i, d2)
        elif d1 > d2 and d2 <= d3 and d2 < E0 - 3*noiseStd:
            if not valleys or i - valleys[-1][0] > 6:  # 第一次遇到谷值或距离上一个谷值超过6个数
                valleys.append((i, d2))
            elif valleys[-1][1] > d2:  # 局部区域有更小的谷值
                valleys[-1] = (i, d2)

    # print('+++++++++\n', peaks)
    # print('---------\n', valleys)
    return peaks, valleys

def compEpp(Edata):
    peaks, valleys = findPeakValley(Edata, E0)
    # 峰值点
    peaks_x = [peak[0] for peak in peaks]
    peaks_y = [peak[1] for peak in peaks]
    # 谷值点
    valleys_x = [valley[0] for valley in valleys]
    valleys_y = [valley[1] for valley in valleys]

    # 计算峰谷值Epp, 并提取稳定段的Epps
    EppSize = min(len(peaks), len(valleys))
    Epp = [peaks[i][1] - valleys[i][1] for i in range(EppSize)]
    print('峰谷对的个数=', EppSize)
    start, end = 20, 200
    # for i in range(start, EppSize):
    #     if Epp[i] / Epp[i - 1] < 1.02:
    #         start = i
    #         break
    # for j in range(start, EppSize):
    #     if Epp[j] / Epp[j - 1] < 0.96:
    #         end = j
    #         break
    Epps = Epp[start: end]
    print('start={}, end={}'.format(start, end))

    mean, std = np.mean(Epps), np.std(Epps)
    Epp_x = np.linspace(mean - 3 * std, mean + 3 * std, 39)
    Epp_y = np.exp(-(Epp_x - mean) ** 2 / (2 * std * std)) / (std * np.sqrt(2 * np.pi))
    # k-s校验,样本大于 300
    # 输出结果中第一个为统计量，第二个为P值（注：统计量越接近0就越表明数据和标准正态分布拟合的越好，
    # 如果P值大于显著性水平，通常是0.05，接受原假设，则判断样本的总体服从正态分布）
    # r = stats.kstest(Epp, 'norm')

    # 正态分布检验 样本量大于20，小于50;
    # 输出结果中第一个为统计量，第二个为P值（注：p值大于显著性水平0.05，认为样本数据符合正态分布）
    r = stats.normaltest(Epps)

    print('r={}, Emean={:.4f}, var={:.4f}'.format(r, mean, std * std))

    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(np.arange(Edata.index.size), Edata['E'])
    plt.plot(peaks_x, peaks_y, '+')
    plt.plot(valleys_x, valleys_y, '*')
    plt.plot([peaks[start][0], peaks[end][0]], [peaks[start][1], peaks[end][1]], 'ro')
    plt.grid()
    ax2 = plt.subplot(1, 2, 2)
    plt.hist(Epps, bins=20, density=True)
    plt.plot(Epp_x, Epp_y)
    plt.show()

if __name__ == '__main__':
    # 用pandas读取
    noiseStd = 0.00001   # 噪声值

    data = pd.read_csv('data.csv', names=['i', 'E'], header=0)
    E0 = data.loc[0: 10000]['E'].mean()    # 求E的均值
    compEpp(data.loc[0: 5000])

    # 对16个线圈进行轮询
    # for i in range(16):
    #     compEpp(data.loc[i * 5000: (i + 1) * 5000])

