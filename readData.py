import binascii
import csv
import re
import sys
import struct
import time
from queue import Queue
from multiprocessing.dummy import Process

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports

from coilArray import CoilArray
from predictorViewer import findPeakValley


def runsend(open=True):
    port = "COM3"
    ser = serial.Serial(port, 921600, timeout=0.3)
    if ser.isOpen():
        print("open {} success!\n".format(port))
    else:
        raise RuntimeError("open failed")

    # 启动发射端的命令
    cmd = "EB 90 01 32 05 01 90 00 00 00 AA 55" if open else "EB 90 00 32 05 01 90 00 00 00 AA 55"
    cmdList = cmd.split(' ')
    cmd2 = b''.join([binascii.a2b_hex(s) for s in cmdList])
    #ser.write(cmd2)

    data = ser.readline()
    '''
    data数据格式
    b'9\xaaU\xeb\x90chC:039\xaaU\xeb\x90chC:039'
    \xaaU\xeb\x90chD:000
    '''
    dataRe = re.findall(b'ch\w:\d{3}', data)
    print('--time:{:.3f}--------data size={}---------'.format(time.time(), len(dataRe)))
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
    
    currents = []
    for ch in chCurrents:
        chCurrent = chCurrents.get(ch)
        currenMax = 0
        for current in chCurrent:
            if chCurrent.get(current) > currenMax:
                currenMax = current 
        currents.append(currenMax)
    print(currents)
    return currents


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


def PIOS_CRC_updateByte(crc, data) :
    crc_table = [
        0x00, 0x07, 0x0e, 0x09, 0x1c, 0x1b, 0x12, 0x15, 0x38, 0x3f, 0x36, 0x31, 0x24, 0x23, 0x2a, 0x2d,
        0x70, 0x77, 0x7e, 0x79, 0x6c, 0x6b, 0x62, 0x65, 0x48, 0x4f, 0x46, 0x41, 0x54, 0x53, 0x5a, 0x5d,
        0xe0, 0xe7, 0xee, 0xe9, 0xfc, 0xfb, 0xf2, 0xf5, 0xd8, 0xdf, 0xd6, 0xd1, 0xc4, 0xc3, 0xca, 0xcd,
        0x90, 0x97, 0x9e, 0x99, 0x8c, 0x8b, 0x82, 0x85, 0xa8, 0xaf, 0xa6, 0xa1, 0xb4, 0xb3, 0xba, 0xbd,
        0xc7, 0xc0, 0xc9, 0xce, 0xdb, 0xdc, 0xd5, 0xd2, 0xff, 0xf8, 0xf1, 0xf6, 0xe3, 0xe4, 0xed, 0xea,
        0xb7, 0xb0, 0xb9, 0xbe, 0xab, 0xac, 0xa5, 0xa2, 0x8f, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9d, 0x9a,
        0x27, 0x20, 0x29, 0x2e, 0x3b, 0x3c, 0x35, 0x32, 0x1f, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0d, 0x0a,
        0x57, 0x50, 0x59, 0x5e, 0x4b, 0x4c, 0x45, 0x42, 0x6f, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7d, 0x7a,
        0x89, 0x8e, 0x87, 0x80, 0x95, 0x92, 0x9b, 0x9c, 0xb1, 0xb6, 0xbf, 0xb8, 0xad, 0xaa, 0xa3, 0xa4,
        0xf9, 0xfe, 0xf7, 0xf0, 0xe5, 0xe2, 0xeb, 0xec, 0xc1, 0xc6, 0xcf, 0xc8, 0xdd, 0xda, 0xd3, 0xd4,
        0x69, 0x6e, 0x67, 0x60, 0x75, 0x72, 0x7b, 0x7c, 0x51, 0x56, 0x5f, 0x58, 0x4d, 0x4a, 0x43, 0x44,
        0x19, 0x1e, 0x17, 0x10, 0x05, 0x02, 0x0b, 0x0c, 0x21, 0x26, 0x2f, 0x28, 0x3d, 0x3a, 0x33, 0x34,
        0x4e, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5c, 0x5b, 0x76, 0x71, 0x78, 0x7f, 0x6a, 0x6d, 0x64, 0x63,
        0x3e, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2c, 0x2b, 0x06, 0x01, 0x08, 0x0f, 0x1a, 0x1d, 0x14, 0x13,
        0xae, 0xa9, 0xa0, 0xa7, 0xb2, 0xb5, 0xbc, 0xbb, 0x96, 0x91, 0x98, 0x9f, 0x8a, 0x8d, 0x84, 0x83,
        0xde, 0xd9, 0xd0, 0xd7, 0xc2, 0xc5, 0xcc, 0xcb, 0xe6, 0xe1, 0xe8, 0xef, 0xfa, 0xfd, 0xf4, 0xf3 ]

    return crc_table[crc ^ data];

def crc8Calculate(curCrc, data):
    val = curCrc
    for d in data:
        val = PIOS_CRC_updateByte(val, d)
    return val


def readRecData(q1, q2, q3):
    '''
    通信协议：
        name：       head/type/size/objid/instid/data/crc
        len(byte) :   1 / 1  / 2  /  4  /   2  / 0~64/ 1
    '''
    UAVTALK_SYNC_VAL = 0x3c
    UAVTALK_TYPE_MASK = 0x78
    UAVTALK_TYPE_VER = 0x20
    UAV_OBJ_ADC = 0xcda3a85c
    UAV_OBJ_ACC = 0x8B7BBFB6
    UAV_OBJ_GYRO = 0xADC3A85C

    port = "COM5"
    ser = serial.Serial(port, 460800, timeout=0.5)
    if ser.isOpen():
        print("open {} success!\n".format(port))
    else:
        raise RuntimeError("open failed")

    adcVlist = []
    coilIndex = 0
    
    while True:
        head = ser.read()

        if len(head) > 0 :
            headVal = int.from_bytes(head,'little')   # 获取head

            if headVal == UAVTALK_SYNC_VAL :
                #crc8_head = PIOS_CRC_updateByte(0, headVal)
                typeh = ser.read()
                dataType = int.from_bytes(typeh, 'little')  # 获取type
                
                if dataType == UAVTALK_TYPE_VER:
                    size = ser.read(2)
                    dataLen = int.from_bytes(size, 'little')   # 获取size
                    #print("dataLen=", dataLen)
                    readLen = dataLen - 4 

                    dataBuff = ser.read(readLen)   # 读取objId+instid+data
                    
                    # crc8 = crc8Calculate(crc8_head, typeh + size + dataBuff)  # crc8校验算法
                    # crc = ser.read()    # 读取crc校验码
                    # if not crc8 == crc[0]:
                    #     print("crc is not right!")

                    objId = int.from_bytes(dataBuff[0: 4], 'little')  # 获取objId
                    #print("objId=", objId)

                    if objId == UAV_OBJ_ADC:
                        instADCId = int.from_bytes(dataBuff[4: 6], 'little')   # 获取ADC instId
                        #print("instADCId=", instADCId)      

                        if instADCId == 0 and adcVlist:
                            coilIndex += 1
                            q1.put(adcVlist.copy())
                            #print("t={:.3f}, adc_num={}, coili={}".format(time.time(), len(adcVlist), coilIndex%16))
                            adcVlist.clear()        

                        for i in range(6, readLen, 2):
                            adcV = int.from_bytes(dataBuff[i: i + 2], 'little') * 1e-6
                            adcVlist.append(adcV)

                        if instADCId == 7:
                            coilIndex += 1
                            q1.put(adcVlist.copy())
                            #print("t={:.3f}, adc_num={}, coili={}".format(time.time(), len(adcVlist), coilIndex%16))
                            adcVlist.clear() 
                               
                    elif objId == UAV_OBJ_GYRO:
                        gyro = struct.unpack('f'*3, dataBuff[6: 18])
                        q2.put(gyro)    
                    
                    elif objId == UAV_OBJ_ACC:
                        acc = struct.unpack('f'*3, dataBuff[6: 18])
                        q3.put(acc)
                    

def plotRecData(qADC, qGyro, qAcc, currents, file=None):
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
    state = np.array([0, 0, 145 + 7.5, 1, 0, 0, 0])  # 真实值
    coils = CoilArray(np.array(currents))
    VppSim = coils.h(state) * 2

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
            adcVmean = np.array(adcV).mean()
            vpp = findPeakValley(adcV, adcVmean, 6e-6)
            if vpp:
                iVS += 1
                xVS.put(iVS)
                yMea.put(vpp * 1e6)
                ySim.put(VppSim[iVS % 16 - 1])
                print('iVS={}, vpp={:.2f}uV'.format(iVS % 16, vpp * 1e6))

            n = len(adcV)
            for v in adcV:
                yADC.put(v - adcVmean)
                iADC += 1
                xADC.put(iADC)

                if fcsv:  # 导出数据
                    fcsv.writerow((iADC, v - adcVmean))

        else:
            n = 500
            for _ in range(n):
                yADC.put(0)
                iADC += 1
                xADC.put(iADC)

                if fcsv:  # 导出数据
                    fcsv.writerow((iADC, 0))
        curveADC.setData(xADC.queue, yADC.queue)

        if iADC > 50000:
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
    timer.start(10)
    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()


def runRec():
    q1, q2, q3 = Queue(), Queue(), Queue()
    #readRecData(q1, q2, q3)
    procReadRec = Process(target=readRecData, args=(q1, q2, q3))
    procReadRec.daemon = True
    procReadRec.start()
    #time.sleep(1)

    #currents = runsend(open=True)
    currents = [2.21, 2.22, 2.31, 2.39, 2.33, 2.31, 2.29, 2.34, 2.29, 2.38, 2.36, 2.31, 2.35, 2.41, 2.42, 2.35]
    plotRecData(q1, q2, q3, currents=currents, file=None)


if __name__ == "__main__":
    runRec()
