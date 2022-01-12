import re
import time
from queue import Queue
from multiprocessing.dummy import Process

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


if __name__ == "__main__":
    q = Queue()
    proReadData = Process(target=readCurrent, args=(q,))
    proReadData.daemon = True
    proReadData.start()

    getData(q)