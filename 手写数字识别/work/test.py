# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
from cnn import Ccnn
import cv2
import gParam

#识别图像可视化
def showimg(label,path, select,result):
    full_path = path + '%d' % select + gParam.FILE_TYPE
    img = cv2.imread(full_path, 1)
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
    if label == result:
        result = str(result) + ' ' + 'Result is right!'
    else:
        result = str(result) + ' ' + 'Result is wrong!'
    cv2.putText(img, result, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    #cv2.imshow('number' + str(label), img)
    savepath = 'test_result/'+str(select)+'_result.png'
    cv2.imwrite(savepath, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cLyNum = 20
pLyNum = 20
fLyNum = 100
oLyNum = 10
test_start = 8000  #测试图片起始位置，共10000幅图
test_num = 1000    #选取测试集个数

myCnn = Ccnn(cLyNum, pLyNum, fLyNum, oLyNum)   #初始化卷积网络
ylabel = myCnn.read_label(gParam.LAB_PATH)     #读取图片对应的真实数字的标签文件

#读取保存好的权重参数
myCnn.kernel_c = np.load('train_result/kernel_c.npy')
myCnn.cLyBias = np.load('train_result/cLyBias.npy')
myCnn.weight_f = np.load('train_result/weight_f.npy')
myCnn.kernel_f = np.load('train_result/kernel_f.npy')
myCnn.fLyBias = np.load('train_result/fLyBias.npy')
myCnn.weight_output = np.load('train_result/weight_output.npy')

rightnum = 0   #正确预测的个数
for i in range(test_start, test_start + test_num):
    data = myCnn.read_pic_data(gParam.TOP_PATH, i)   #读取单张图片28*28

    ylab = int(ylabel[i])      #读取此图片对应的真实数字
    d_m, d_n = shape(data)     #读取图片大小
    m_c = d_m - gParam.C_SIZE + 1     #卷积结果特征图大小
    n_c = d_n - gParam.C_SIZE + 1     #28-5+1=24,24*24
    m_p = int(m_c / myCnn.pSize)      #池化结果大小
    n_p = int(n_c / myCnn.pSize)      #24/2=12,12*12
    state_c = zeros((m_c, n_c, myCnn.cLyNum))   #初始化卷积结果：24*24*20
    state_p = zeros((m_p, n_p, myCnn.pLyNum))   #初始化池化结果：12*12*20
    for n in range(myCnn.cLyNum):
        state_c[:, :, n] = myCnn.convolution(data, myCnn.kernel_c[:, :, n])  #卷积
        tmp_bias = ones((m_c, n_c)) * myCnn.cLyBias[:, n]
        state_c[:, :, n] = np.tanh(state_c[:, :, n] + tmp_bias)  # 加上偏置项然后过激活函数
        state_p[:, :, n] = myCnn.pooling(state_c[:, :, n], myCnn.pooling_a)  #池化
    state_f, state_f_pre = myCnn.convolution_f1(state_p, myCnn.kernel_f, myCnn.weight_f)  #全连接结果1*1*100

    # 进入激活函数
    state_fo = zeros((1, myCnn.fLyNum))  #全连接层经过激活函数的结果1*100
    for n in range(myCnn.fLyNum):
        state_fo[:, n] = np.tanh(state_f[:, :, n] + myCnn.fLyBias[:, n])  #a+b
    # 进入softmax层
    output = myCnn.softmax_layer(state_fo)  #1*10
    # 预测结果
    y_pre = output.argmax(axis=1)

    print('真实数字为', ylab, '预测数字是', y_pre)

    if (i >= test_start and i < (test_start + 30)) or ylab != y_pre:    #保存30个识别结果以及错误结果
        showimg(ylab,gParam.TOP_PATH,i,y_pre)
    if ylab == y_pre:
        rightnum = rightnum + 1        #如果识别正确，+1

print('手写数字识别准确率为：',rightnum * 100 / test_num,'%.')