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


def findPeak(data):
    dataSize = len(data)
    dataPeak = []
    for i in range(dataSize-2):
        d1, d2, d3 = eval(data[i][1]), eval(data[i+1][1]), eval(data[i+2][1])
        if not (d1 < d2 < d3 or d1 > d2 > d3):
            dataPeak.append(data[i+1])
    return dataPeak

if __name__ == '__main__':
    # 用csv读取
    # data = []
    # with open('data.csv', 'r') as f:
    #     f_csv = csv.reader(f)
    #     i = 0
    #     for row in f_csv:
    #         data.append((row[0], row[1]))
    #         i += 1

    # 用pandas读取

    data = pd.read_csv('data.csv', names=['t', 'E'], header=0)
    x = np.array(data['t'][5000:6000])
    y = np.array(data['E'][5000:6000])
    #
    # peaks, _ = signal.find_peaks(y, distance=3)   # 较为耗时，不能用于实时采集
    # print(peaks)
    # x_peaks = [x[i] for i in peaks]
    # y_peaks = [y[i] for i in peaks]

    # peaks2 = signal.find_peaks_cwt(y, np.arange(1,10))  # 适用于毛刺较多时的寻峰
    # x_peaks2 = [x[i] for i in peaks2]
    # y_peaks2 = [y[i] for i in peaks2]

    # 峰值点
    x_pos_slice = x[21::25]
    y_pos_slice = y[21::25]
    # 谷值点
    x_neg_slice = x[33::25]
    y_neg_slice = y[33::25]

    peaks_num = min(x_pos_slice.size, x_neg_slice.size)
    print("peaks_num=", peaks_num)
    Epp = np.zeros(peaks_num)
    for i in range(peaks_num):
        Epp[i] = y_pos_slice[i] - y_neg_slice[i]

    mean, std = np.mean(Epp), np.std(Epp)
    Epp_x = np.linspace(mean - 3 * std, mean + 3 * std, 39)
    Epp_y = np.exp(-(Epp_x - mean) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    # k-s校验,样本大于 300
    #输出结果中第一个为统计量，第二个为P值（注：统计量越接近0就越表明数据和标准正态分布拟合的越好，
    #如果P值大于显著性水平，通常是0.05，接受原假设，则判断样本的总体服从正态分布）
    #r = stats.kstest(Epp, 'norm')

    # 正态分布检验 样本量大于20，小于50;
    # 输出结果中第一个为统计量，第二个为P值（注：p值大于显著性水平0.05，认为样本数据符合正态分布）
    r = stats.normaltest(Epp)
    print('r={}, mean={}, var={}'.format(r, mean, std * std))

    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.plot(x_pos_slice, y_pos_slice, '+')
    plt.plot(x_neg_slice, y_neg_slice, '*')
    plt.grid()

    ax2 = plt.subplot(1, 2, 2)
    plt.hist(Epp, bins=20, density=True)
    plt.plot(Epp_x, Epp_y)
    plt.show()