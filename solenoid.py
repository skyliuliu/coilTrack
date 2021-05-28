# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/5/12 9:08
desc: 螺线管建模计算磁场和感应电压
'''
import math
import time

import numpy as np

def solenoid(n1=200, nr1=8, n2=100, nr2=2, r1=5, d1=0.6, r2=2.5, d2=0.05, ii=2, freq=20000, d=(0, 0, 0.2 - 0.0075),
              em1=(0, 0, 1), em2=(0, 0, 1)):
    """
    基于毕奥-萨法尔定律，计算发射线圈在接收线圈中产生的感应电动势
    :param n1: 发射线圈匝数 [1]
    :param nr1: 发射线圈层数 [1]
    :param n2: 接收线圈匝数 [1]
    :param nr2: 接收线圈层数 [1]
    :param r1: 发射线圈内半径 [mm]
    :param d1: 发射线圈线径 [mm]
    :param r2: 接收线圈内半径 [mm]
    :param d2: 接收线圈线径 [mm]
    :param ii: 激励电流的幅值 [A]
    :param freq: 激励信号的频率 [Hz]
    :param d: 初级线圈中心到次级线圈中心的位置矢量 [m]
    :param em1: 发射线圈的朝向 [1]
    :param em2: 接收线圈的朝向 [1]
    :return E: 感应电压 [1e-6V]
    """
    nh = int(n1 / nr1)
    ntheta = 100
    theta = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
    r = np.linspace(r1, r1 + nr1 * d1, nr1, endpoint=False)
    h = np.linspace(0, nh * d1, nh, endpoint=False)
    hh = np.array([[0, 0, hi] for hi in h])

    drxy = np.stack((np.cos(theta), np.sin(theta), np.zeros(ntheta)), 1)  # 电流元在xy平面的位置方向
    dlxy = np.stack((-np.sin(theta), np.cos(theta), np.zeros(ntheta)), 1)  # 电流元在xy平面的电流方向
    dlxy = np.vstack([dlxy] * nh)

    dr = np.zeros((ntheta * n1, 3), dtype=np.float)
    dl = np.zeros((ntheta * n1, 3), dtype=np.float)
    for i in range(nr1):
        dl[ntheta * nh * i: ntheta * nh * (i + 1), :] = r[i] * 2 * math.pi / ntheta * dlxy
        for j in range(nh):
            dr[ntheta * (i * nh + j): ntheta * (i * nh + j + 1), :] = r[i] * drxy + hh[j]

    er = np.array(d) * 1000 - dr
    rNorm = np.linalg.norm(er, axis=1, keepdims=True)
    er0 = er / rNorm
    dB = 1e-4 * ii * np.cross(dl, er0) / rNorm ** 2
    B = dB.sum(axis=0)

    # 精确计算线圈的面积，第i层线圈的面积为pi * (r + d * i) **2
    S2 = n2 // nr2 * math.pi * sum([(r2 + d2 * k) ** 2 for k in range(nr2)]) / 1000000
    E = 2 * math.pi * freq * S2 * np.dot(B, em2) * 1e6
    return E

if __name__ == '__main__':
    t = time.time()
    E = 1.0
    for i in range(100):
        E = solenoid()
    print('E={:.2f}, t={:.2f}'.format(E, (time.time() - t) * 1000))