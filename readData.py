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
        #print(chCurrents)
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
        # print(data)
        if data.startswith(b'GYRO'):
            gyroData = str(data[6: -3], encoding='utf-8').split(' ')
            gyro_w = [float(w) for w in gyroData]
            t2 = time.time()
            print('gyroscope: time={:.0f}ms'.format((t2 - t0) * 1000))
            qGyro.put({t2: gyro_w})

        elif data.startswith(b'ACCDATA'):
            accData = str(data[9: -3], encoding='utf-8').split(' ')
            acc_a = [float(a) for a in accData]
            t3 = time.time()
            print('accelerator: time={:.0f}ms'.format((t3 - t0) * 1000))
            qAcc.put({t3: acc_a})

        else:
            adcRe = re.findall(b' \d{5}', data)
            if adcRe:
                adcV = np.array([int(v) / 1e7 for v in adcRe])  # 原始信号放大1000倍，然后在MCU中放大10000倍
                t1 = time.time()
                print('ADC: time={:.0f}ms, size={}'.format((t1 - t0) * 1000, len(adcV)))
                qADC.put({t1: adcV})

        

def getRecData(qADC, qGyro, qAcc):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="采样板信号")
    win.resize(1000, 750)
    win.setWindowTitle("接收端采样")
    pg.setConfigOptions(antialias=True)

    pADC = win.addPlot(title='ADC', colspan=2)
    pADC.addLegend()
    pADC.setLabel('left', '电压', units='V')
    pADC.setLabel('bottom', 'points', units='1')
    pADC.showGrid(x=True, y=True)
    curveADC = pADC.plot()
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
    
    xADC = Queue()
    yADC = Queue()
    xGyro = Queue()
    xAcc = Queue()
    qGyro_x = Queue()
    qGyro_y = Queue()
    qGyro_z = Queue()
    qAcc_x = Queue()
    qAcc_y = Queue()
    qAcc_z = Queue()
    fs = 20000   # ADC采样率
    dt = 1 / fs
    def update():
        # ADC  
        if not qADC.empty():
            ts = list(qADC.get().keys())[0]
            adcV = list(qADC.get().values())[0]
            n = len(adcV)
            for i in range(n):
                yADC.put(adcV[i])
                xADC.put(ts + i * dt)

        curveADC.setData(xADC.queue, yADC.queue)
        # if iADC > 20000:
        #     for _ in range(n):
        #         xADC.get()
        #         yADC.get()    
        
        # gyroscope
        if not qGyro.empty():
            ts = list(qGyro.get().keys())[0]
            w = list(qGyro.get().values())[0]
            qGyro_x.put(w[0])
            qGyro_y.put(w[1])
            qGyro_z.put(w[2])
            xGyro.put(ts)

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
            ts = list(qAcc.get().keys())[0]
            a = list(qAcc.get().values())[0]
            qAcc_x.put(a[0])
            qAcc_y.put(a[1])
            qAcc_z.put(a[2])
            xAcc.put(ts)

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

            
if __name__ == "__main__":
    q1, q2, q3 = Queue(), Queue(), Queue()
    # readRecData(q1, q2, q3)
    procReadRec = Process(target=readRecData, args=(q1, q2, q3))
    procReadRec.daemon = True
    procReadRec.start()

    getRecData(q1, q2, q3)