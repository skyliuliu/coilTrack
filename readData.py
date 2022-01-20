import csv
import re
import sys
import time
from queue import Queue
from multiprocessing.dummy import Process

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import serial
import serial.tools.list_ports

from dataTool import findPeakValley
from calculatorLM import Tracker


def readCurrent(q):
    port = "COM3"
    ser = serial.Serial(port, 921600, timeout=0.5)
    if ser.isOpen():
        print("open {} success!\n".format(port))
    else:
        raise RuntimeError("open failed")

    while True:
        data = ser.readline()
        '''
        data数据格式
        b'9\xaaU\xeb\x90chC:039\xaaU\xeb\x90chC:039'
        \xaaU\xeb\x90chD:000
        '''
        dataRe = re.findall(b'ch\w:\d{3}', data)
        print('--time:{:.3f}--------data size={}---------'.format(time.time(),len(dataRe)))
        chCurrents = {
            'ch1': {},
            'ch2': {},
            'ch3': {},
            'ch4': {},
            'ch5': {},
            'ch6': {},
            'ch7': {},
            'ch8': {},
            'ch9': {},
            'chA': {},
            'chB': {},
            'chC': {},
            'chD': {},
            'chE': {},
            'chF': {},
            'chG': {}
        }
        for chc in dataRe:
            ch = str(chc[:3], encoding='utf-8')
            currentStr = str(chc[-3:], encoding='utf-8')
            # 取整数部分和小数部分，拼接起来后只取小数点后两位有效数字
            current = round(int(currentStr[0]) + int(currentStr[1:3]) * 0.01, 2)  
            if chCurrents[ch].get(current) == None:
                chCurrents[ch][current] = 0
            else:
                chCurrents[ch][current] += 1
        print(chCurrents)
        q.put(chCurrents)

def getData(q):
    '''
    从队列中提取数据
    '''
    while True:
        if not q.empty():
            data = q.get()
            for key in data.keys():
                chCurrents = data.get(key)
                currents = list(chCurrents.keys())
                if currents == []:
                    break
                currentSum = 0
                currentNum = 0
                for current in currents:
                    if current != 0:
                        currentSum += current * chCurrents.get(current)
                        currentNum += chCurrents.get(current)
                data[key] = round(currentSum / currentNum, 2) if currentNum else 0.0
            print(data)
        time.sleep(0.5)

def readRecData(qADC, qGyro, qAcc):
    port = "COM5"
    ser = serial.Serial(port, 921600, timeout=0.5)
    if ser.isOpen():
        print("open {} success!\n".format(port))
    else:
        raise RuntimeError("open failed")

    while True:
        t0 = time.time()
        
        data = ser.readline()
        adcRe = re.findall(b' \d{5}', data)
        if adcRe:
            adcV = np.array([int(v) / 1e7 for v in adcRe])  # 原始信号放大1000倍，然后在MCU中放大10000倍
            adcAvg = adcV.mean()
            qADC.put(adcV - adcAvg)
            
        else:
            qADC.put([0] * 200)
            
            gyroRe = re.search(b'GYRO: (.*)', data)  
            if gyroRe:
                gyroData = re.findall(b'(-?\d*\.\d*)', gyroRe.group()[:-2])
                qGyro.put([float(w) for w in  gyroData])

            accRe = re.search(b'ACCDATA: (.*)', data)
            if accRe:
                accData = re.findall(b'(-?\d*\.\d*)', accRe.group()[:-2])
                qAcc.put([float(a) for a in  accData])

        
def plotRecData(qADC, qGyro, qAcc, file=None):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="采样板信号")
    win.resize(1000, 750)
    win.setWindowTitle("接收端采样")
    pg.setConfigOptions(antialias=True)

    pADC = win.addPlot(title='ADC', col=0)
    pADC.addLegend()
    pADC.setLabel('left', '电压', units='V')
    pADC.setLabel('bottom', 'points', units='1')
    pADC.showGrid(x=True, y=True)
    curveADC = pADC.plot()

    pMeaVSsim = win.addPlot(title='实测 vs 理论', col=1)
    pMeaVSsim.addLegend()
    pMeaVSsim.setLabel('left', '电压', units='uV')
    pMeaVSsim.setLabel('bottom', '采样包数', units='1')
    curveMea = pMeaVSsim.plot(pen='r', name='实测')
    curveSim = pMeaVSsim.plot(pen='g', name='理论')
    win.nextRow()
    
    pGyro = win.addPlot(title='gyroscope', col=0)
    pGyro.addLegend()
    pGyro.setLabel('left', '角速度', units='deg/s')
    pGyro.setLabel('bottom', 'points', units='1')
    pGyro.showGrid(x=True, y=True)
    curveGyro_x = pGyro.plot(pen='r', name='x')
    curveGyro_y = pGyro.plot(pen='g', name='y')
    curveGyro_z = pGyro.plot(pen='b', name='z')

    pAcc = win.addPlot(title='accelerator', col=1)
    pAcc.addLegend()
    pAcc.setLabel('left', '加速度', units='mg')
    pAcc.setLabel('bottom', 'points', units='1')
    pAcc.showGrid(x=True, y=True)
    curveAcc_x = pAcc.plot(pen='r', name='x')
    curveAcc_y = pAcc.plot(pen='g', name='y')
    curveAcc_z = pAcc.plot(pen='b', name='z')
    
    # 导出数据
    if file:
        f = open(file, 'w', newline='')
        fcsv = csv.writer(f)
    else:
        fcsv = None
    
    # ADC
    xADC = Queue()
    yADC = Queue()
    iADC = 0
    xVS = Queue()
    yMea = Queue()
    ySim = Queue()
    iVS = 0

    # sim data
    state = np.array([0, 0, 0.21 - 0.0075, 1, 0, 0, 0])  # 真实值
    tracker = Tracker(state)
    VppSim = tracker.h(state) * 2

    # IMU
    xGyro = Queue()
    xAcc = Queue()
    qGyro_x = Queue()
    qGyro_y = Queue()
    qGyro_z = Queue()
    qAcc_x = Queue()
    qAcc_y = Queue()
    qAcc_z = Queue()
    xIMU = 0
    def update():
        nonlocal iADC, xIMU, iVS
        # ADC  
        if not qADC.empty():
            adcV = qADC.get()
            vpp = findPeakValley(adcV, 0, 4e-6)
            if vpp:
                iVS += 1
                xVS.put(iVS)
                yMea.put(vpp * 1e6)
                ySim.put(VppSim[iVS%16 - 1])
                print('vpp={}uV'.format(vpp * 1e6))

            n = len(adcV)
            for v in adcV:
                yADC.put(v)
                iADC += 1
                xADC.put(iADC)

                if fcsv:   # 导出数据
                    fcsv.writerow((iADC, v))    
                
        else:
            n = 500
            for _ in range(n):
                yADC.put(0)
                iADC += 1
                xADC.put(iADC)

                if fcsv:   # 导出数据
                    fcsv.writerow((iADC, 0))
        curveADC.setData(xADC.queue, yADC.queue)
        
        if iADC > 100000:
            for _ in range(n):
                xADC.get()
                yADC.get() 

        if xVS.qsize() > 16:
            xVS.get()
            yMea.get()
            ySim.get()
        curveMea.setData(xVS.queue, yMea.queue)
        curveSim.setData(xVS.queue, ySim.queue)
        
        # gyroscope
        xIMU += 1
        if not qGyro.empty():
            w = qGyro.get()
            qGyro_x.put(w[0])
            qGyro_y.put(w[1])
            qGyro_z.put(w[2])
            xGyro.put(xIMU)

            curveGyro_x.setData(xGyro.queue, qGyro_x.queue)
            curveGyro_y.setData(xGyro.queue, qGyro_y.queue)
            curveGyro_z.setData(xGyro.queue, qGyro_z.queue)

        if xGyro.qsize() > 100:
            xGyro.get()
            qGyro_x.get()
            qGyro_y.get()
            qGyro_z.get()

        # accelerator
        if not qAcc.empty():
            a = qAcc.get()
            qAcc_x.put(a[0])
            qAcc_y.put(a[1])
            qAcc_z.put(a[2])
            xAcc.put(xIMU)

            curveAcc_x.setData(xAcc.queue, qAcc_x.queue)
            curveAcc_y.setData(xAcc.queue, qAcc_y.queue)
            curveAcc_z.setData(xAcc.queue, qAcc_z.queue)

        if xAcc.qsize() > 100:
            xAcc.get()
            qAcc_x.get()
            qAcc_y.get()
            qAcc_z.get()
        
    timer = pg.Qt.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()

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
    dataSize = len(data)
    #startIndex = data._stat_axis._start
    # 找出满足条件1的峰和谷
    peaks, valleys = [], []
    for i in range(1, dataSize-1):
        d1, d2, d3 = data[i-1], data[i], data[i+1]   # 用于实时获取的数据
        point = (i+1, d2)
        if d1 < d2 and d2 >= d3 and d2 > E0 + 3*noiseStd:
            if not peaks or i - peaks[-1][0] > 9:  # 第一次遇到峰值或距离上一个峰值超过9个数
                peaks.append(point)
            elif peaks[-1][1] < d2:   # 局部区域有更大的峰值
                peaks[-1] = point
        elif d1 > d2 and d2 <= d3 and d2 < E0 - 3*noiseStd:
            if not valleys or i - valleys[-1][0] > 9:  # 第一次遇到谷值或距离上一个谷值超过9个数
                valleys.append(point)
            elif valleys[-1][1] > d2:  # 局部区域有更小的谷值
                valleys[-1] = point

    peaks_y = [peak[1] for peak in peaks]
    valleys_y = [valley[1] for valley in valleys]

    peakMean = sum(peaks_y) / len(peaks_y) if len(peaks_y) else 0
    valleyMean = sum(valleys_y) / len(valleys_y) if len(valleys_y) else 0
    return peakMean - valleyMean

def readRec():
    q1, q2, q3 = Queue(), Queue(), Queue()
    # readRecData(q1, q2, q3)
    procReadRec = Process(target=readRecData, args=(q1, q2, q3))
    procReadRec.daemon = True
    procReadRec.start()

    
    plotRecData(q1, q2, q3, file=None)

def readSend():
    q = Queue()
    readCurrent(q)


if __name__ == "__main__":
    readRec()