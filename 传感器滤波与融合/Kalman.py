import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import pandas as pd
import math

#——————————————————————————————————读取数据————————————————————————————————————————#
def readfile(file):
    Radar_data = []
    Lidar_data = []
    datainput = pd.read_csv(file)
    datainput = np.mat(datainput)
    m, n = datainput.shape

    for i in range(m):
        if datainput[i, 0] == 'R':     #Radar主要数据为测量(ρ,ψ,ρ˙)，实际(x,y,vx,vy)
            Radar_data.append([float(datainput[i, 2]), float(datainput[i, 3]), float(datainput[i, 4]), \
                              float(datainput[i, 6]), float(datainput[i, 7]), float(datainput[i, 8]),
                              float(datainput[i, 9])])
        else:                          #Lidar主要数据为测量(x,y),实际(x,y,vx,vy)
            Lidar_data.append([float(datainput[i, 2]), float(datainput[i, 3]), float(datainput[i, 6]), \
                              float(datainput[i, 7]), float(datainput[i, 8]), float(datainput[i, 9])])

    Radar_data = np.mat(Radar_data)
    Lidar_data = np.mat(Lidar_data)
    print(Radar_data.shape,Lidar_data.shape)
    return Radar_data,Lidar_data

#——————————————————————————————————限制一个角在[−π,π]————————————————————————————————#
def control_psi(psi):
    while (psi > np.pi or psi < -np.pi):
        if psi > np.pi:
            psi = psi - 2 * np.pi
        if psi < -np.pi:
            psi = psi + 2 * np.pi
    return psi
#————————————————————————————————————状态转移矩阵————————————————————————————————————#
def getphi(x,dt):
    transition = lambda y: np.vstack((
        y[0] + (y[2] / y[4]) * (np.sin(y[3] + y[4] * dt) - np.sin(y[3])),
        y[1] + (y[2] / y[4]) * (-np.cos(y[3] + y[4] * dt) + np.cos(y[3])),
        y[2],
        y[3] + y[4] * dt,
        y[4]))

    # when omega is 0
    transition0 = lambda m: np.vstack((
        m[0] + m[2] * np.cos(m[3]) * dt,
        m[1] + m[2] * np.sin(m[3]) * dt,
        m[2],
        m[3] + m[4] * dt,
        m[4]))

    # 定义状态转移函数的雅克比矩阵（函数,输入参数为上一步state list）：
    J_A = nd.Jacobian(transition)
    J_A_1 = nd.Jacobian(transition0)

    if np.abs(x[4, 0]) < 0.0001:  # 如果omega是0, 那么处于直线行驶状态
        x = transition0(x.ravel().tolist())
        x[3, 0] = control_psi(x[3, 0])
        JA = J_A_1(x.ravel().tolist())
    else:
        x = transition(x.ravel().tolist())
        x[3, 0] = control_psi(x[3, 0])
        JA = J_A(x.ravel().tolist())
    return JA,x

#——————————————————————————————————测量函数——————————————————————————————————————#
# 定义测量函数：
measurement_function = lambda k: np.vstack((np.sqrt(k[0] * k[0] + k[1] * k[1]),
                                            math.atan2(k[1], k[0]),
                                            (k[0] * k[2] * np.cos(k[3]) + k[1] * k[2] * np.sin(k[3])) / np.sqrt(
                                            k[0] * k[0] + k[1] * k[1])))
# 定义测量函数的雅克比矩阵（函数,输入参数为当前步state list）：
J_H = nd.Jacobian(measurement_function)

#——————————————————————————————————————获取tao————————————————————————————————#
def gettao(x,dt):
    tao = np.zeros([5, 2])
    tao[0, 0] = 0.5 * dt * dt * np.cos(x[3, 0])
    tao[1, 0] = 0.5 * dt * dt * np.sin(x[3, 0])
    tao[2, 0] = dt
    tao[3, 1] = 0.5 * dt * dt
    tao[4, 1] = dt
    return tao

#————————————————————————————————数据保存————————————————————————————————————————#
px = []
py = []
pxl = []
pyl = []
pxr = []
pyr = []

rx = []
ry = []

mxL = []
myL = []
mxR = []
myR = []

def savestates(s,sl,sr, gx, gy, ml1, ml2,mr1,mr2):
    #Kalman滤波融合预测结果
    px.append(s[0])
    py.append(s[1])
    # Kalman滤波Lidar预测结果
    pxl.append(sl[0])
    pyl.append(sl[1])
    # Kalman滤波Radar预测结果
    pxr.append(sr[0])
    pyr.append(sr[1])
    #实际位置
    rx.append(gx)
    ry.append(gy)
    #Lidar测量结果
    mxL.append(ml1)
    myL.append(ml2)
    # Radar测量结果
    mxR.append(mr1)
    myR.append(mr2)

#————————————————————————————————————————绘制滤波结果图————————————————————————————————————————#
def printfig():
    #绘制实际位置、Lidar的测量结果以及EKF滤波预测结果
    plt.figure(figsize=(25, 25), dpi=125)
    plt.plot(rx, ry, 'ro', label="real")
    plt.plot(mxL, myL, 'go', label="mLidar")
    plt.plot(pxl, pyl, 'bo', label="EKFLidar")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('resultL', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('resultL.png')
    plt.show()

    # 绘制实际位置、Radar的测量结果以及EKF滤波预测结果
    plt.figure(figsize=(25, 25), dpi=125)
    plt.plot(rx, ry, 'ro', label="real")
    plt.plot(mxR, myR, 'go', label="mRadar")
    plt.plot(pxr, pyr, 'bo', label="EKFRadar")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('resultR', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('resultR.png')
    plt.show()

    # 绘制实际位置、Lidar和Radar各自的EKF滤波预测结果，以及融合结果
    plt.figure(figsize=(25, 25), dpi=125)
    plt.plot(rx, ry, 'ro', label="real")
    plt.plot(pxl, pyl, 'yo', label="EKFRidar")
    plt.plot(pxr, pyr, 'go', label="EKFRadar")
    plt.plot(px, py, 'bo', label="EKFfusion")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('result', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('result.png')
    plt.show()
#——————————————————————————————————求列表的差值——————————————————————————————#
def error(list1,list2):
     e = []
     x,ii=np.shape(np.mat(list1))
     for i in range (ii):
         t = list1[i] - list2 [i]
         e.append(t)
     return e
#——————————————————————————————————————绘制误差结果图——————————————————————————————————————#
ex = []
ey = []
exl = []
eyl = []
exr = []
eyr = []
exml = []
eyml = []
exmr = []
eymr = []
xx = []
def printerror():
    # Kalman滤波融合结果误差
    ex = error(px, rx)
    ey = error(py, ry)
    # Kalman滤波Lidar预测结果误差
    exl = error(pxl, rx)
    eyl = error(pyl, ry)
    # Kalman滤波Radar预测结果误差
    exr = error(pxr, rx)
    eyr = error(pyr, ry)
    # Lidar测量结果误差
    exml = error(mxL, rx)
    eyml = error(myL, ry)
    # Radar测量结果误差
    exmr = error(mxR, rx)
    eymr = error(myR, ry)
    x, ii = np.shape(np.mat(ex))
    for i in range(ii):
        xx.append(i)
    # 绘制Lidar和Radar各自的EKF滤波预测结果误差，以及融合结果误差
    #x方向
    plt.figure(figsize=(20, 10), dpi=125)
    plt.plot(xx, ex, 'r', label="errorx")
    plt.plot(xx, exl, 'y', label="errorxl")
    plt.plot(xx, exr, 'b', label="errorxr")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('errorx', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('errorx.png')
    plt.show()
    #y方向
    plt.figure(figsize=(20, 10), dpi=125)
    plt.plot(xx, ey, 'r', label="errory")
    plt.plot(xx, eyl, 'y', label="erroryl")
    plt.plot(xx, eyr, 'b', label="erroryr")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('errory', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('errory.png')
    plt.show()

    # 绘制LidarEKF滤波预测结果误差，测量结果误差
    #x方向
    plt.figure(figsize=(20, 10), dpi=125)
    plt.plot(xx, exl, 'r', label="errorxl")
    plt.plot(xx, exml, 'g', label="errorxml")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('errorLidarx', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('errorLidarx.png')
    plt.show()
    #y方向
    plt.figure(figsize=(20, 10), dpi=125)
    plt.plot(xx, eyl, 'r', label="erroryl")
    plt.plot(xx, eyml, 'g', label="erroryml")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('errorLidary', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('errorLidary.png')
    plt.show()

    # 绘制RadarEKF滤波预测结果误差，测量结果误差
    #x方向
    plt.figure(figsize=(20, 10), dpi=125)
    plt.plot(xx, exr, 'r', label="errorxr")
    plt.plot(xx, exmr, 'g', label="errorxmr")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('errorRadarx', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('errorRadarx.png')
    plt.show()
    #y方向
    plt.figure(figsize=(20, 10), dpi=125)
    plt.plot(xx, eyr, 'r', label="erroryr")
    plt.plot(xx, eymr, 'g', label="errorymr")

    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)  # 坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('errorRadary', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('errorRadary.png')
    plt.show()

#求均方误差
def rmse(estimates, actual):
    result = np.sqrt(np.mean((estimates-actual)**2))
    return result

#——————————————————————————————————————————主函数————————————————————————————————————————#
if __name__=='__main__':
    filename = "Radar_Lidar_Data1.csv"
    Radardata,Lidardata=readfile(filename)   #读取Radar和Lidar数据

    PL = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])  # 初始化P
    PR = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])  # 初始化P
    H_Lidar = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])  # 初始化Lidar观测矩阵HL
    H_Radar = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])  # 初始化Radar观测矩阵HR
    #H_Radar = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],[0., 0., 1., 0., 0.]])  # 初始化Radar观测矩阵HR
    R_Lidar = np.array([[0.0225, 0.], [0., 0.0225]])  # 初始化Lidar测量噪声
    R_Radar = np.array([[0.0225, 0.], [0., 0.0225]])  # 初始化Radar测量噪声
    #R_Radar = np.array([[0.09, 0., 0.], [0., 0.0009, 0.], [0., 0., 0.09]])  # 初始化Radar测量噪声
    std_noise_a = 2.0  # 处理噪声的中直线加速度项的标准差 σa
    std_noise_yaw_w = 0.3  # 转角加速度项的标准差σω˙

    stateL = np.zeros(5)    #设定初始状态为Lidar的初始x，y
    stateL[0] = Lidardata[0,0]
    stateL[1] = Lidardata[0,1]
    stateL = stateL.reshape([5, 1])
    stateR = np.zeros(5)  # 设定初始状态为Lidar的初始x，y
    stateR[0] = Radardata[0, 0]*np.cos(control_psi(Radardata[0, 1]))
    stateR[1] = Radardata[0, 0]*np.sin(control_psi(Radardata[0, 1]))
    stateR = stateR.reshape([5, 1])

    num1,num2=Lidardata.shape
    dt=0.2
    I = np.eye(5)

    for k in range (num1):
        measurement_L = Lidardata[k,:]  #读取Lidar第i行数据
        measurement_R = Radardata[k, :]  #读取Radar第i行数据
        xL_m = Lidardata[k, 0]
        yL_m = Lidardata[k, 1]
        rouR_m = Radardata[k, 0]
        psiR_m = Radardata[k, 1]
        psidR_m = Radardata[k, 1]
        xR_m = rouR_m*np.cos(control_psi(psiR_m))
        yR_m = rouR_m*np.sin(control_psi(psiR_m))
        #观测结果Z
        ZL = np.array([[xL_m],[yL_m]])
        #ZR = np.array([[rouR_m],[psiR_m],[psidR_m]])
        ZR = np.array([[xR_m], [yR_m]])

        #实际位置
        x_real = (Lidardata[k, 2] + Radardata[k, 3]) / 2
        y_real = (Lidardata[k, 3] + Radardata[k, 4]) / 2
        vx_real = (Lidardata[k, 4] + Radardata[k, 5]) / 2
        vy_real = (Lidardata[k, 5] + Radardata[k, 6]) / 2

        #雅克比矩阵
        JAL,stateL = getphi(stateL, dt)
        JAR,stateR = getphi(stateR, dt)
        #处理噪声的协方差矩阵Q
        taoL=gettao(stateL,dt)
        taoR=gettao(stateR,dt)
        Q = np.diag([std_noise_a * std_noise_a, std_noise_yaw_w * std_noise_yaw_w])
        QL = np.dot(np.dot(taoL, Q), taoL.T)
        QR = np.dot(np.dot(taoR, Q), taoR.T)
        #预测P
        PL = np.dot(np.dot(JAL, PL),JAL.T)+QL
        PR = np.dot(np.dot(JAR, PR), JAR.T) + QR

        #更新LidarR,K,状态，P
        RL = np.dot(np.dot(H_Lidar, PL), H_Lidar.T) + R_Lidar
        KL = np.dot(np.dot(PL, H_Lidar.T), np.linalg.inv(RL))
        yL = ZL - np.dot(H_Lidar, stateL)
        yL[1, 0] = control_psi(yL[1, 0])
        stateL = stateL + np.dot(KL, yL)
        stateL[3, 0] = control_psi(stateL[3, 0])
        PL = np.dot((I - np.dot(KL, H_Lidar)), PL)

        # 更新RadarR,K,状态，P
        RR = np.dot(np.dot(H_Radar, PR), H_Radar.T) + R_Radar
        KR = np.dot(np.dot(PR, H_Radar.T), np.linalg.inv(RR))
        '''
        JH = J_H(stateR.ravel().tolist())
        RR = np.dot(np.dot(JH, PR), JH.T) + R_Radar
        KR = np.dot(np.dot(PR, JH.T), np.linalg.inv(RR))
        
        sss = measurement_function(stateR.ravel().tolist())
        if np.abs(sss[0, 0]) < 0.0001:
            sss[2, 0] = 0
        yR = ZR - sss
        '''
        yR = ZR - np.dot(H_Radar, stateR)
        yR[1, 0] = control_psi(yR[1, 0])
        stateR = stateR + np.dot(KR, yR)
        stateR[3, 0] = control_psi(stateR[3, 0])
        #PR = np.dot((I - np.dot(KR, JH)), PR)
        PR = np.dot((I - np.dot(KR, H_Radar)), PR)

        #数据融合X=PP1niX1+PP2niX2
        PL_1=np.linalg.inv(PL)    #求逆
        PR_1=np.linalg.inv(PR)
        P=np.linalg.inv(PL_1+PR_1)
        state=np.dot(P,(np.dot(PL_1,stateL)+np.dot(PR_1,stateR)))
        print('k=',k)
        savestates(state.ravel().tolist(), stateL.ravel().tolist(),stateR.ravel().tolist(),\
                   x_real, y_real, xL_m, yL_m, xR_m, yR_m)

    printfig()    #显示滤波结果
    printerror()   #误差比较
    print('Lidar测量结果均方误差:','\nx:',rmse(np.array(mxL), np.array(rx)),
          '\ny:',rmse(np.array(myL), np.array(ry)),
          '\nLidar滤波预测结果均方误差:','\nx:',rmse(np.array(pxl), np.array(rx)),
          '\ny:',rmse(np.array(pyl), np.array(ry)),
          '\nRadar测量结果均方误差:','\nx:',rmse(np.array(mxR), np.array(rx)),
          '\ny:',rmse(np.array(myR), np.array(ry)),
          '\nRadar滤波预测结果均方误差:','\nx:',rmse(np.array(pxr), np.array(rx)),
          '\ny:',rmse(np.array(pyr), np.array(ry)),
          '\n滤波融合预测结果均方误差:','\nx:',rmse(np.array(px), np.array(rx)),
          '\ny:',rmse(np.array(py), np.array(ry)))
