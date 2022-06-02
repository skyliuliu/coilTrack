# coilTrack

## 1. 简介
   　　基于python3的线圈阵列磁定位方法，使用外置的线圈阵列发射AC交变磁场，在胶囊内置线圈中产生感应信号，实现对胶囊的定位。

## 2. 硬件
- 发射线圈：至少9个空心线圈，形成阵列排布
- 接收线圈：一个单轴的空心线圈
- 驱动电路：控制发射线圈依次工作
- 采样电路：提取接收线圈的感应电压

## 3. 文件组成

| 文件名              | 简介                           |
|:-------------------|:-------------------------------|
|agreement.py        | 读取接收端的串口通讯协议 |
| calculatorLM.py    | 使用LM算法实现的定位方法      |
| calculatorUKF.py   | 使用UKF算法实现的定位方法     |
| predictorViewer.py | 绘图工具，包括定位过程，误差分布等 |
| readData.py        | 实时读取接收端的数据 |
| coilArray.py       | 定义发射线圈和接收线圈参数的类    |
| dataTool.py        | 时域数据处理函数，包含寻峰、FFT、绘图等|
| Lie.py             | 李代数的实现 |
| se3LM.py           | 基于李代数+LM算法实现的定位方法 |
| predictor.py       | 定位程序的主文件，calculatorLM + se3LM的集成 |
| data.csv           | 采集的时域数据 |
| measureData.csv    | 每个线圈产生的信号幅值 |
| requirements.txt    | 依赖包及版本号 |

## 4. 软件安装方法
- python >= 3.8
- pip工具升级到最新版本：```python3 -m pip install --upgrade pip```
- 依赖包的版本要求如requirements.txt所示，或者用最新版本
- 依赖包安装方法：pip install -r requirements.txt

## 5. 使用方法
#### 5.1 读取实时数据
   　　运行```readData.py```，出现UI界面后通过串口工具向发射端发送启动命令

#### 5.2 实时定位
   　　运行```predictor.py```，出现UI界面后通过串口工具向发射端发送启动命令
