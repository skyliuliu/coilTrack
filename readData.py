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
        adcRe = re.findall(b' \d{5}', data)
        if adcRe:
            adcV = np.array([int(v) / 1e7 for v in adcRe])  # 原始信号放大1000倍，然后在MCU中放大10000倍
            # adcAvg = adcV.mean()
            qADC.put(adcV)
            #print('time={:.3f}, adc size={}'.format(time.time(), len(adcV)))
            gyroRe = re.search(b'GYRO: (.*)', data)  
            if gyroRe:
                gyroData = re.findall(b'(-?\d*\.\d*)\t', gyroRe.group())
                qGyro.put([float(w) for w in  gyroData])
            print('1-----------dt={:.3f}s------------'.format(time.time() - t0))
        else:
            qADC.put([2.5e-3] * 200)
            
            accRe = re.search(b'ACCDATA: (.*)', data)
            if accRe:
                accData = re.findall(b'(-?\d*\.?\d*)\t', accRe.group())
                qAcc.put([float(a) for a in  accData])
            print('2-----------dt={:.3f}s------------'.format(time.time() - t0))


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
    iADC = 0
    iGyro = 0
    iAcc = 0
    def update():
        nonlocal iADC, iGyro, iAcc
        # ADC  
        if not qADC.empty():
            adcV = qADC.get()
            n = len(adcV)
            for v in adcV:
                yADC.put(v)
                iADC += 1
                xADC.put(iADC)
        else:
            n = 500
            for _ in range(n):
                yADC.put(0)
                iADC += 1
                xADC.put(iADC)
        curveADC.setData(xADC.queue, yADC.queue)
        # if iADC > 20000:
        #     for _ in range(n):
        #         xADC.get()
        #         yADC.get()    
        
        # gyroscope
        if not qGyro.empty():
            iGyro += 1
            w = qGyro.get()
            qGyro_x.put(w[0])
            qGyro_y.put(w[1])
            qGyro_z.put(w[2])
            xGyro.put(iGyro)

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
            iAcc += 1
            a = qAcc.get()
            qAcc_x.put(a[0])
            qAcc_y.put(a[1])
            qAcc_z.put(a[2])
            xAcc.put(iAcc)

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