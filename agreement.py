'''
通信协议：
    name：       head/type/size/objid/instid/timestamp/data/crc
    len(byte) :   1 / 1  / 2(从head-crc) /  4   /   2  / 2(该位上发不存在) / 0~64/ 1
'''
import serial
from multiprocessing.dummy import Process
import time
import struct
import ctypes
import binascii
import numpy as np

UAVTALK_SYNC_VAL = 0x3c
UAVTALK_TYPE_MASK = 0x78
UAVTALK_TYPE_VER = 0x20
UAV_OBJ_SENSOR = 0xcda3a85c

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
    for i in range(len(data)) :
        val = PIOS_CRC_updateByte(val, data[i])
    return val

def sensorUnpack(data):
    # print(data)
    lsmSensorData = np.array([[1,2,3,4],[5,5,5,5],[1,2,3,4],[1,2,3,4],[5,5,5,5],[1,2,3,4]], dtype = 'float32')
    accel_x = lsmSensorData[0]
    accel_y = lsmSensorData[1]
    accel_z = lsmSensorData[2]
    gyro_x = lsmSensorData[3]
    gyro_y = lsmSensorData[4]
    gyro_z = lsmSensorData[5]
    akMag1SensorData = np.array([[1,2,3,4],[5,5,5,5],[1,2,3,4]], dtype = 'int16')
    mag1_x = akMag1SensorData[0]
    mag1_y = akMag1SensorData[1]
    mag1_z = akMag1SensorData[2]
    akMag2SensorData = np.array([[1,2,3,4],[5,5,5,5],[1,2,3,4]], dtype = 'int16')
    mag2_x = akMag1SensorData[0]
    mag2_y = akMag1SensorData[1]
    mag2_z = akMag1SensorData[2]
    sensorReadTimestamp = np.array([[1,2,3,4]], dtype = 'float32')
    timedata = sensorReadTimestamp[0]

    objId = int.from_bytes(data[0:4], 'little')   # 获取objId
    # print(objId)
    if objId == UAV_OBJ_SENSOR :
        instId = int.from_bytes(data[4:6], 'little')   # 获取instId
        for i in range(4) :
            buff = struct.unpack('<f', data[6+i*4:10+i*4])
            accel_x[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<f', data[22+i*4:26+i*4])
            accel_y[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<f', data[38+i*4:42+i*4])
            accel_z[i] = np.asarray(buff)

        for i in range(4) :
            buff = struct.unpack('<f', data[54+i*4:58+i*4])
            gyro_x[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<f', data[70+i*4:74+i*4])
            gyro_y[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<f', data[86+i*4:90+i*4])
            gyro_z[i] = np.asarray(buff)
        # print(struct.unpack('<h', data[102:104]))
        for i in range(4) :
            buff = struct.unpack('<h', data[102+i*2:104+i*2])
            mag1_x[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<h', data[110+i*2:112+i*2])
            mag1_y[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<h', data[118+i*2:120+i*2])
            mag1_z[i] = np.asarray(buff)

        for i in range(4) :
            buff = struct.unpack('<h', data[126+i*2:128+i*2])
            mag2_x[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<h', data[134+i*2:136+i*2])
            mag2_y[i] = np.asarray(buff)
        for i in range(4) :
            buff = struct.unpack('<h', data[142+i*2:144+i*2])
            mag2_z[i] = np.asarray(buff)

        for i in range(4) :
            buff = struct.unpack('<f', data[150+i*4:154+i*4])
            timedata[i] = np.asarray(buff)
        print(data[150:154])
        print("accel_x" ,accel_x)
        print("accel_y", accel_y)
        print("accel_z", accel_z)
        print("gyro_x" ,gyro_x)
        print("gyro_y", gyro_y)
        print("gyro_z", gyro_z)
        print("mag1_x" ,mag1_x)
        print("mag1_y", mag1_y)
        print("mag1_z", mag1_z)
        print("mag2_x" ,mag2_x)
        print("mag2_y", mag2_y)
        print("mag2_z", mag2_z)
        print("timedata", timedata)


def receive(mark,serial_port):

    # Wait a second to let the port initialize
    time.sleep(1)

    while True:
        data = serial_port.read()

        if len(data) > 0 :
            syncVal = int.from_bytes(data,'little')   # 获取head

            if syncVal == UAVTALK_SYNC_VAL :
                crc8 = PIOS_CRC_updateByte(0, syncVal)
                dataType = int.from_bytes(serial_port.read(),'little')  # 获取type
                #print(dataType)
                if dataType == UAVTALK_TYPE_VER:
                    crc8 = PIOS_CRC_updateByte(crc8, dataType)
                    dataLen = int.from_bytes(serial_port.read(2),'little')   # 获取size
                    print("dataLen=", dataLen)
                    _dataLen = ctypes.c_short(dataLen)
                    high_8 = (_dataLen.value & 0xff00) >> 8
                    crc8 = PIOS_CRC_updateByte(crc8, high_8)
                    low_8 = (_dataLen.value & 0x00ff)
                    crc8 = PIOS_CRC_updateByte(crc8, low_8)
                    readLen = dataLen + 4 + 2 + 1
                    dataBuff = serial_port.read(readLen)
                    objId = int.from_bytes(dataBuff[0:4], 'little')  # 获取objId

                    if len(dataBuff) > 0 and objId == UAV_OBJ_SENSOR:
                        crc8 = crc8Calculate(crc8, dataBuff)
                        crc8Val = dataBuff[-1]
                        decData = binascii.b2a_hex(dataBuff).decode('utf-8')
                        print(decData)
                        print(decData[-10:-8])
                        print("declen", len(decData[:-12]))
                        print("declen",len(decData[:-12]))
                        print(int.from_bytes(dataBuff[-2:-6], 'little'))
                        print("crc8",crc8)
                        print("crc8Val", crc8Val)

                        sensorUnpack(dataBuff)



if __name__ == '__main__':
    serial_port = serial.Serial('COM5', 230400, timeout=0.5)

    if serial_port.isOpen() :
        print("open success")
    else :
        print("open failed")

    # Wait a second to let the port initialize
    mark = 0
    receive(mark, serial_port)


