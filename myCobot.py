# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2022/2/24 15:38
desc: 大象机器人的控制程序
'''
import time

from pymycobot.mycobot import MyCobot
from pymycobot.genre import Coord


mc = MyCobot('COM17', 115200)
if mc.is_power_on() == None:
    mc.power_on()

pos1 = mc.get_coords()
print("pos1=", pos1)

mc.send_coord(Coord.X.value, pos1[0] + 100, 2)
#mc.send_coords([-57.4, -248.0, 188.2, -152.73, 0.8, -177.55], 10, 0)

time.sleep(10)

pos2 = mc.get_coords()
print("pos2=", pos2)
print("pos2 - pos1 = ", [round(x - y, 1) for x, y in zip(pos1, pos2)])
