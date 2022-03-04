# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/7/4 16:41
desc: 用于实测数据的处理
'''
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats
from scipy.fftpack import fft


def findPeakValley(data, E0, noiseStd):
    '''
    寻峰算法:
    1、对连续三个点a, b, c，若a<b && b>c，且b>E0，则b为峰点；若a>b && b<c，且b<E0，则b为谷点
    2、保存邻近的峰点和谷点，取x=±9个点内的最大或最小值作为该区段的峰点或谷点
    :param data: 【pd】读取的原始数据
    :param E0: 【float】原始数据的平均值
    :param noiseStd: 【float】噪声值
    :param Vpp: 【float】原始数据的平均峰峰值
    :return:
    '''
    dataSize = len(data)    # 数据的长度
    startIndex = data._stat_axis.start   # 数据的起始下标
    # 找出满足条件的峰和谷
    peaks, valleys = [], []
    for i in range(1, dataSize-1):
        d1, d2, d3 = data['E'][i-1+startIndex], data['E'][i+startIndex], data['E'][i+1+startIndex]   # 当data为通过pandas导入的数据
        #d1, d2, d3 = data[i-1], data[i], data[i+1]   # 用于实时获取的数据
        point = (i + startIndex, d2)
        if d1 < d2 and d2 >= d3 and d2 > E0 + noiseStd:
            if not peaks or i + startIndex - peaks[-1][0] > 9:  # 第一次遇到峰值或距离上一个峰值超过9个数
                peaks.append(point)
            elif peaks[-1][1] < d2:   # 局部区域有更大的峰值
                peaks[-1] = point
        elif d1 > d2 and d2 <= d3 and d2 < E0 - noiseStd:
            if not valleys or i + startIndex - valleys[-1][0] > 9:  # 第一次遇到谷值或距离上一个谷值超过9个数
                valleys.append(point)
            elif valleys[-1][1] > d2:  # 局部区域有更小的谷值
                valleys[-1] = point

    # 计算每包数据的峰和谷的平均值
    peakMeans = []  # 存储每包数据的peakMean值
    start = peaks[0][0]
    peakSum = peaks[0][1]
    index = 1
    for point in peaks[1:]:
        if point[0] - start < 2000:   # 每包数据起始和结尾下标之差不超过800，下同
            peakSum += point[1]
            index += 1
        else:
            peakMean = peakSum / index
            peakMeans.append(peakMean)
            start = point[0]
            peakSum = point[1]
            index = 1

    valleyMeans = []  # 存储每包数据的peakMean值
    start = valleys[0][0]
    valleySum = valleys[0][1]
    index = 1
    for point in valleys[1:]:
        if point[0] - start < 2000:
            valleySum += point[1]
            index += 1
        else:
            valleyMean = valleySum / index
            valleyMeans.append(valleyMean)
            start = point[0]
            valleySum = point[1]
            index = 1
    print("peakMeans=", np.round(peakMeans, 6))
    print("valleyMeans=", np.round(valleyMeans, 6))
    print("vmMeans=", (np.round(peakMeans, 6) - np.round(valleyMeans, 6)) / 2)

    # plot peaks and valleys in data fig
    peaks_x = [peak[0] for peak in peaks]
    peaks_y = [peak[1] for peak in peaks]
    valleys_x = [valley[0] for valley in valleys]
    valleys_y = [valley[1] for valley in valleys]
    plt.plot(data['i'], data['E'])
    plt.plot(peaks_x, peaks_y, '+')
    plt.plot(valleys_x, valleys_y, '*')
    plt.show()

    return peaks, valleys
    

def compEpp(Edata):
    peaks, valleys = findPeakValley(Edata, E0, noiseStd=2e-5)
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

def compFFT(data):
    '''
    对实测的时域数据进行傅里叶变换，提取f0和对应的幅值
    :param data: 实测结果
    :return:
    '''
    dataE = data.E
    peaks = []

    p = []   # 保存非零的数据
    PG = False   # 启动筛选的开关
    for v in dataE:
        if v:    # 当遇到非零的数据时启动开关
            PG = True
        elif PG and v == 0:   # 数据包筛选完成后直接退出
            fftPack(p)
            PG = False
            p = []
        if PG:    # 筛选数据
            p.append(v)

def fftPack(p):
    '''
    对每个包中的数据实时傅里叶变换
    :param p: 每个包的数据（非零）
    :return:
    '''
    print("pL=", len(p))
    pack = p[200:]   # 选取稳定阶段的数据
    L = len(pack)   # 数据长度
    print("L=", L)
    N = int(np.power(2, np.ceil(np.log2(L))))  # 下一个最近二次幂
    Fs = 964 * 100    # 采样率

    FFT_y1 = np.abs(fft(pack, N)) / L * 2   # N点FFT 变化,但处于信号长度
    FFT_y1 = FFT_y1[range(int(N / 2))]   # 取一半
    freq = np.arange(int(N / 2)) * Fs / N   # 频率坐标

    peak = FFT_y1.max()    # 提取最大值
    f0 = freq[FFT_y1.tolist().index(peak)]    # 提取f0

    plt.plot(freq, FFT_y1)
    plt.xlabel("f/Hz")
    plt.ylabel("v/V")
    plt.scatter(f0, peak, color='red', marker='*')
    plt.text(f0+100, peak, "f0={:.0f}Hz, peak={:.2e}V".format(f0, peak))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # 用pandas读取
    data = pd.read_csv('adcV_20ms.csv', names=['i', 'E'], header=0)
    E0 = data.loc[0: 1000]['E'].mean()  # 求E的均值

    # 寻峰，并计算均值
    findPeakValley(data, E0, noiseStd=6e-6)

    # compEpp(data.loc[0: 65000])

    #compFFT(data)

