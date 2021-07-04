# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/7/4 16:41
desc: 用于实测数据E的处理
'''
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def findPeak(data):
    dataSize = len(data)
    dataPeak = []
    for i in range(dataSize-2):
        d1, d2, d3 = eval(data[i][1]), eval(data[i+1][1]), eval(data[i+2][1])
        if not (d1 < d2 < d3 or d1 > d2 > d3):
            dataPeak.append(data[i+1])
    return dataPeak

if __name__ == '__main__':
    data = []
    with open('data.csv', 'r') as f:
        f_csv = csv.reader(f)
        i = 0
        for row in f_csv:
            data.append((row[0], row[1]))
            i += 1

    x = [eval(i[0]) for i in data]
    y = [eval(i[1]) for i in data]
    num_peak = signal.find_peaks(y, distance=5)
    x_peaks = [eval(data[i][0]) for i in num_peak[0]]
    y_peaks = [eval(data[i][1]) for i in num_peak[0]]
    print(num_peak)

    plt.plot(x, y)
    plt.plot(x_peaks, y_peaks, '+')
    plt.grid()
    plt.show()