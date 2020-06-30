# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
from cnn import Ccnn
import math
import cv2
import gParam

# 设定初始参数
cLyNum = 20       #卷积层卷积核个数
pLyNum = 20       #池化层节点个数
fLyNum = 100      #全连接层卷积核个数
oLyNum = 10       #输出层10个节点对应0-9，10个数字
train_num = 8000   #训练数据取得图像个数

myCnn = Ccnn(cLyNum, pLyNum, fLyNum, oLyNum)         #初始化卷积网络
ylabel = myCnn.read_label(gParam.LAB_PATH)           #读取标签
for iter0 in range(gParam.MAX_ITER_NUM):
    for i in range(train_num):
        print(iter0, ":", i)
        data = myCnn.read_pic_data(gParam.TOP_PATH, i)       #在第iter0次迭代中读取第i幅图像
        ylab = int(ylabel[i])            #读取第i幅图像对应的标签
        d_m, d_n = shape(data)           #图像大小
        m_c = d_m - gParam.C_SIZE + 1    #卷积结果大小
        n_c = d_n - gParam.C_SIZE + 1
        m_p = int(m_c / myCnn.pSize)     #池化结果大小
        n_p = int(n_c / myCnn.pSize)
        state_c = zeros((m_c, n_c, myCnn.cLyNum))        #初始化卷积结果:24*24*20
        state_p = zeros((m_p, n_p, myCnn.pLyNum))        #初始化池化结果:12*12*20
        for n in range(myCnn.cLyNum):
            state_c[:, :, n] = myCnn.convolution(data, myCnn.kernel_c[:, :, n])    #对图像用第n个卷积核卷积，得到卷积结果的第n幅特征图
            tmp_bias = ones((m_c, n_c)) * myCnn.cLyBias[:, n]        #偏置项
            state_c[:, :, n] = np.tanh(state_c[:, :, n] + tmp_bias)  # 加上偏置项然后过激活函数tanh
            state_p[:, :, n] = myCnn.pooling(state_c[:, :, n], myCnn.pooling_a)   #卷积结果的第n幅特征图进行池化
        state_f, state_f_pre = myCnn.convolution_f1(state_p, myCnn.kernel_f, myCnn.weight_f)   #全连接层1*1*100,12*12*100

        # 进入激活函数
        state_fo = zeros((1, myCnn.fLyNum))  #全连接层经过激活函数的结果1*100
        for n in range(myCnn.fLyNum):
            state_fo[:, n] = np.tanh(state_f[:, :, n] + myCnn.fLyBias[:, n])
        # 进入softmax层
        output = myCnn.softmax_layer(state_fo)    #1*10

        # 预测结果
        y_pre = output.argmax(axis=1)
        #更新权重
        myCnn.cnn_upweight(ylab, data, state_c, state_p, state_fo, state_f_pre, output)

#存储训练结果参数
np.save('train_result/kernel_c.npy', myCnn.kernel_c)   #5*5*20
np.save('train_result/cLyBias.npy', myCnn.cLyBias)     #1*20
np.save('train_result/weight_f.npy', myCnn.weight_f)   #20*100
np.save('train_result/kernel_f.npy', myCnn.kernel_f)   #12*12*100
np.save('train_result/fLyBias.npy', myCnn.fLyBias)     #1*100
np.save('train_result/weight_output.npy', myCnn.weight_output)  #100*10
print('参数保存成功!!!')