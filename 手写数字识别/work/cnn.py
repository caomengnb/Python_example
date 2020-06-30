# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
import math
import gParam
import copy
import scipy.signal as signal

# 生成形状为*args（行*列）的数组，其中的元素取均匀分布在a，b之间的随机数
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

# 创建Cnn类
class Ccnn:
    #初始化各项参数
    def __init__(self, cLyNum, pLyNum, fLyNum, oLyNum):
        self.cLyNum = cLyNum        #初始化卷积层卷积核个数
        self.pLyNum = pLyNum        #初始化池化层节点个数
        self.fLyNum = fLyNum        #初始化全连接层卷积核个数
        self.oLyNum = oLyNum        #初始化输出层10个节点对应0-9，10个数字
        self.pSize = gParam.P_SIZE  #初始化池化层核的大小
        self.yita = 0.01         #学习速率
        self.cLyBias = rand_arr(-0.1, 0.1, 1, cLyNum)       #初始化卷积层偏置项
        self.fLyBias = rand_arr(-0.1, 0.1, 1, fLyNum)       #初始化全连接层偏置项
        self.kernel_c = zeros((gParam.C_SIZE, gParam.C_SIZE, cLyNum))   #初始化cLyNum个尺寸为C_SIZE*C_SIZE的卷积核
        self.kernel_f = zeros((gParam.F_SIZE, gParam.F_SIZE, fLyNum))     #初始化fLyNum个尺寸为F_SIZE*F_SIZE的全连接层卷积核
        for i in range(cLyNum):             #设定卷积核
            self.kernel_c[:, :, i] = rand_arr(-0.1, 0.1, gParam.C_SIZE, gParam.C_SIZE)
        for i in range(fLyNum):             #设定全连接层卷积核
            self.kernel_f[:, :, i] = rand_arr(-0.1, 0.1, gParam.F_SIZE, gParam.F_SIZE)
        self.pooling_a = ones((self.pSize, self.pSize)) / (self.pSize ** 2)   #初始化池化层卷积核
        self.weight_f = rand_arr(-0.1, 0.1, pLyNum, fLyNum)     #初始化全连接层权重
        self.weight_output = rand_arr(-0.1, 0.1, fLyNum, oLyNum)   #初始化输出权重

    #读取图片数据集
    def read_pic_data(self, path, i):
        # print 'read_pic_data'
        data = np.array([])
        full_path = path + '%d' % i + gParam.FILE_TYPE
        try:
            data = mgimg.imread(full_path)  # data is np.array
            data = (double)(data)
            print("读取图片：",'%d'%i,gParam.FILE_TYPE)
        except IOError:
            raise Exception('open file error in read_pic_data():', full_path)
        return data

    #读取标签数据集,存入列表ylab中
    def read_label(self, path):
        # print 'read_label'
        ylab = []
        try:
            fobj = open(path, 'r')
            for line in fobj:
                ylab.append(line.strip())
            fobj.close()
        except IOError:
            raise Exception('open file error in read_label():', path)
        return ylab

    # 卷积层:输入(28*28,5*5),输出(24*24)
    def convolution(self, data, kernel):
        data_row, data_col = shape(data)      #图片尺寸
        kernel_row, kernel_col = shape(kernel)     #卷积核尺寸

        n = data_col - kernel_col + 1       #卷积结果尺寸
        m = data_row - kernel_row + 1
        state = zeros((m, n))         #卷积结果
        for i in range(m):
            for j in range(n):
                temp = multiply(data[i:i + kernel_row, j:j + kernel_col], kernel)  #data的对应位置与卷积核点乘
                state[i][j] = temp.sum()   #state[i][j]为点乘结果所有元素之和
        return state

    # 池化层:输入(24*24,2*2),输出(12*12)
    def pooling(self, data, pooling_a):
        data_r, data_c = shape(data)
        p_r, p_c = shape(pooling_a)

        r0 = int(data_r / p_r)     #池化结果尺寸
        c0 = int(data_c / p_c)
        state = zeros((r0, c0))    #池化结果
        for i in range(r0):
            for j in range(c0):
                temp = multiply(data[p_r * i:p_r * i + p_r, p_c * j:p_c * j + p_c], pooling_a)   #***************
                state[i][j] = temp.sum()    #对应位置与池化核相同大小的区域与池化核点乘，再将所有元素求和

        return state

    # 全连接层:输入(12*12*20,12*12*100,20*100),输出(1*1*100,12*12*100)
    def convolution_f1(self, state_p1, kernel_f1, weight_f1):
        n_p0, n_f = shape(weight_f1)  #n_p0=20(是特征图的个数);n_f是100(全连接层神经元个数)
        m_p, n_p, pCnt = shape(state_p1)  #这个矩阵是池化结果，pCnt个m_p*n_p的特征图
        m_k_f1, n_k_f1, fCnt = shape(kernel_f1)  #全连接层卷积核，fCnt个m_k_f1*n_k_f1矩阵
        state_f1_temp = zeros((m_p, n_p, n_f))   #初始化一个中间量
        state_f1 = zeros((m_p - m_k_f1 + 1, n_p - n_k_f1 + 1, n_f))   #初始化全连接层卷积结果
        for n in range(n_f):
            count = 0
            for m in range(n_p0):
                temp = state_p1[:, :, m] * weight_f1[m][n]
                count = count + temp
            state_f1_temp[:, :, n] = count
            state_f1[:, :, n] = self.convolution(state_f1_temp[:, :, n], kernel_f1[:, :, n])

        return state_f1, state_f1_temp        #1*1*100 / 12*12*100

    # softmax层:输入(1*100,*100*10),输出(1*10)
    def softmax_layer(self, state_f1):
        output = zeros((1, self.oLyNum))     #初始化输出矩阵1*10
        t1 = (exp(np.dot(state_f1, self.weight_output))).sum()  #总体
        for i in range(self.oLyNum):
            t0 = exp(np.dot(state_f1, self.weight_output[:, i]))
            output[:, i] = t0 / t1  #占总体的比例
        return output     #输出为10个预测结果对应的权重

    # 误差反向传播更新权值
    def cnn_upweight(self, ylab, train_data, state_c1, state_s1, state_f1, state_f1_temp, output):
        m_data, n_data = shape(train_data)    #图像大小

        label = zeros((1, self.oLyNum))
        label[:, ylab] = 1    #如果识别精确确，对应的权重应当为1
        delta_layer_output = output - label     #输出误差
        weight_output_temp = copy.deepcopy(self.weight_output)    #输出权重
        delta_weight_output_temp = zeros((self.fLyNum, self.oLyNum))    #输出权重误差

        # 更新输出权重weight_output
        for n in range(self.oLyNum):
            delta_weight_output_temp[:, n] = delta_layer_output[:, n] * state_f1
        weight_output_temp = weight_output_temp - self.yita * delta_weight_output_temp

        # 更新全连接层的偏置fLyBias和卷积核kernel_f
        delta_layer_f1 = zeros((1, self.fLyNum))
        delta_bias_f1 = zeros((1, self.fLyNum))
        delta_kernel_f1_temp = zeros(shape(state_f1_temp))
        kernel_f_temp = copy.deepcopy(self.kernel_f)
        for n in range(self.fLyNum):
            count = 0
            for m in range(self.oLyNum):
                count = count + delta_layer_output[:, m] * self.weight_output[n, m]
            delta_layer_f1[:, n] = np.dot(count, (1 - np.tanh(state_f1[:, n]) ** 2))
            delta_bias_f1[:, n] = delta_layer_f1[:, n]
            delta_kernel_f1_temp[:, :, n] = delta_layer_f1[:, n] * state_f1_temp[:, :, n]

        self.fLyBias = self.fLyBias - self.yita * delta_bias_f1    #1*100
        kernel_f_temp = kernel_f_temp - self.yita * delta_kernel_f1_temp

        # 更新全连接层权重weight_f
        delta_layer_f1_temp = zeros((gParam.F_SIZE, gParam.F_SIZE, self.fLyNum))
        delta_weight_f1_temp = zeros(shape(self.weight_f))
        weight_f1_temp = copy.deepcopy(self.weight_f)
        for n in range(self.fLyNum):
            delta_layer_f1_temp[:, :, n] = delta_layer_f1[:, n] * self.kernel_f[:, :, n]
            #delta_layer_f1_temp[:, :, n] = np.multiply(delta_layer_f1[:, n] ,self.kernel_f[:, :, n])
        for n in range(self.pLyNum):
            for m in range(self.fLyNum):
                temp = delta_layer_f1_temp[:, :, m] * state_s1[:, :, n]
                delta_weight_f1_temp[n, m] = temp.sum()
        weight_f1_temp = weight_f1_temp - self.yita * delta_weight_f1_temp

        # 更新卷积层偏置cLyBias
        n_delta_c = m_data - gParam.C_SIZE + 1
        delta_layer_p = zeros((gParam.F_SIZE, gParam.F_SIZE, self.pLyNum))
        delta_layer_c = zeros((n_delta_c, n_delta_c, self.pLyNum))
        delta_bias_c = zeros((1, self.cLyNum))
        for n in range(self.pLyNum):
            count = 0
            for m in range(self.fLyNum):
                count = count + delta_layer_f1_temp[:, :, m] * self.weight_f[n, m]
            delta_layer_p[:, :, n] = count
            # print shape(np.kron(delta_layer_p[:,:,n], ones((2,2))/4))
            delta_layer_c[:, :, n] = np.kron(delta_layer_p[:, :, n], ones((2, 2)) / 4) \
                                     * (1 - np.tanh(state_c1[:, :, n]) ** 2)
            delta_bias_c[:, n] = delta_layer_c[:, :, n].sum()

        self.cLyBias = self.cLyBias - self.yita * delta_bias_c    #1*20

        # 更新卷积层卷积核 kernel_c
        delta_kernel_c1_temp = zeros(shape(self.kernel_c))
        for n in range(self.cLyNum):
            temp = delta_layer_c[:, :, n]
            r1 = list(map(list, zip(*temp[::1])))  # 逆时针旋转90度
            r2 = list(map(list, zip(*r1[::1])))  # 再逆时针旋转90度
            temp = signal.convolve2d(train_data, r2, 'valid')
            temp1 = list(map(list, zip(*temp[::1])))
            delta_kernel_c1_temp[:, :, n] = list(map(list, zip(*temp1[::1])))
        self.kernel_c = self.kernel_c - self.yita * delta_kernel_c1_temp  #5*5*20
        self.weight_f = weight_f1_temp    #20*100
        self.kernel_f = kernel_f_temp     #12*12*100
        self.weight_output = weight_output_temp   #100*10

    # predict
    def cnn_predict(self, data):
        return