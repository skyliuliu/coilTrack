import re

import serial
import serial.tools.list_ports


def readCurrent():
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
        chCurrents = []
        for chc in dataRe:
            ch = str(chc[:3], encoding='utf-8')
            currentStr = str(chc[-3:], encoding='utf-8')
            current = int(currentStr[0]) + int(currentStr[1:3]) * 0.01
            chCurrents.append({ch: current})
        print(chCurrents)
        print('----------data size={}---------'.format(len(chCurrents)))


if __name__ == "__main__":
    readCurrent()