# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#————————————————————————————————————————半圆环随机点生成——————————————————————————————————————#
num = 500
i_set = np.arange(0, num, 1)
def rangepoint(flag):
    t = np.random.uniform(0.01,0.99,size=num) * np.pi*flag  #随机生成500个θ角，flag=1为上半圆环，flag=-1为下半圆环
    x = np.cos(t)
    y = np.sin(t)
    for i in i_set:
        len = np.sqrt(np.random.uniform(0.16, 1))    #随机生成一个ρ
        x[i] = x[i] * len * 10                  #随机点x坐标
        y[i] = y[i] * len * 10                  #随机点y坐标
    return x,y

#————————————————————————————————————————生成训练集——————————————————————————————————————#
def trainproduce(d):
    plt.figure(figsize=(10, 10), dpi=125)

    x1,y1=rangepoint(1)            #上半圆环随机点生成
    plt.plot(x1, y1, 'bo',label="class=1")          #绘制上半圆环的点
    #——————————————————————————————绘制上半圆环边界————————————————————————————————#
    _t1 = np.arange(0, 3.15, 0.01)
    _x1 = np.cos(_t1) * 10
    _y1 = np.sin(_t1) * 10
    plt.plot(_x1, _y1, 'b-')

    _t2 = np.arange(0, 3.15, 0.01)
    _x2 = np.cos(_t2) * 4
    _y2 = np.sin(_t2) * 4
    plt.plot(_x2, _y2, 'b-')

    _x3 = np.arange(4, 10.1, 0.1)
    _y3 = [0] * 61
    plt.plot(_x3, _y3, 'b-')

    _x4 = np.arange(-10, -3.9, 0.1)
    _y4 = [0] * 61
    plt.plot(_x4, _y4, 'b-')

    x2, y2 = rangepoint(-1) # 下半圆环随机点生成
    x2 = x2 + [d] * 500
    y2 = y2 - [d] * 500
    plt.plot(x2, y2, 'go',label="class=-1")  # 绘制下半圆环的点for i in i_set:
    # ——————————————————————————————绘制下半圆环边界————————————————————————————————#
    _t1 = np.arange(-3.14, 0.01, 0.01)
    _x1 = np.cos(_t1) * 10 + [d] * 315
    _y1 = np.sin(_t1) * 10 - [d] * 315
    plt.plot(_x1, _y1, 'g-')

    _t2 = np.arange(-3.14, 0.01, 0.01)
    _x2 = np.cos(_t2) * 4 + [d] * 315
    _y2 = np.sin(_t2) * 4 - [d] * 315
    plt.plot(_x2, _y2, 'g-')

    _x3 = np.arange(4, 10.1, 0.1) + [d] * 61
    _y3 = [0 - d] * 61
    plt.plot(_x3, _y3, 'g-')

    _x4 = np.arange(-10, -3.9, 0.1) + [d] * 61
    _y4 = [0 - d] * 61
    plt.plot(_x4, _y4, 'g-')

    plt.xlim(-13, 13)         #坐标轴范围
    plt.ylim(-13, 13)
    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)           #坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('train', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('train.png')
    plt.show()
    return x1,y1,x2,y2

#————————————————————————————————————————生成测试集——————————————————————————————————————#
def testproduce(d):
    plt.figure(figsize=(10, 10), dpi=125)

    x11,y11=rangepoint(1)            #上半圆环随机点生成
    plt.plot(x11, y11, 'bo', label="class=1")          #绘制上半圆环的点
    #——————————————————————————————绘制上半圆环边界————————————————————————————————#
    _t1 = np.arange(0, 3.15, 0.01)
    _x1 = np.cos(_t1) * 10
    _y1 = np.sin(_t1) * 10
    plt.plot(_x1, _y1, 'b-')

    _t2 = np.arange(0, 3.15, 0.01)
    _x2 = np.cos(_t2) * 4
    _y2 = np.sin(_t2) * 4
    plt.plot(_x2, _y2, 'b-')

    _x3 = np.arange(4, 10.1, 0.1)
    _y3 = [0] * 61
    plt.plot(_x3, _y3, 'b-')

    _x4 = np.arange(-10, -3.9, 0.1)
    _y4 = [0] * 61
    plt.plot(_x4, _y4, 'b-')

    x22, y22 = rangepoint(-1) # 下半圆环随机点生成
    x22 = x22 + [d] * 500
    y22 = y22 - [d] * 500
    plt.plot(x22, y22, 'go', label="class=1")  # 绘制下半圆环的点
    # ——————————————————————————————绘制下半圆环边界————————————————————————————————#
    _t1 = np.arange(-3.14, 0.01, 0.01)
    _x1 = np.cos(_t1) * 10 + [d] * 315
    _y1 = np.sin(_t1) * 10 - [d] * 315
    plt.plot(_x1, _y1, 'g-')

    _t2 = np.arange(-3.14, 0.01, 0.01)
    _x2 = np.cos(_t2) * 4 + [d] * 315
    _y2 = np.sin(_t2) * 4 - [d] * 315
    plt.plot(_x2, _y2, 'g-')

    _x3 = np.arange(4, 10.1, 0.1) + [d] * 61
    _y3 = [0 - d] * 61
    plt.plot(_x3, _y3, 'g-')

    _x4 = np.arange(-10, -3.9, 0.1) + [d] * 61
    _y4 = [0 - d] * 61
    plt.plot(_x4, _y4, 'g-')

    plt.xlim(-13, 13)         #坐标轴范围
    plt.ylim(-13, 13)
    plt.xticks(fontsize=30)  # 坐标刻度字体大小
    plt.yticks(fontsize=30)
    plt.xlabel('x', fontsize=30)           #坐标轴标签
    plt.ylabel('y', fontsize=30)
    plt.title('test', fontsize=30)
    plt.grid(True)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23, }
    plt.legend(prop=font1)  # 显示图例并设置标签
    plt.savefig('test.png')
    plt.show()
    return x11,y11,x22,y22

#————————————————————————————————————————将样本添加到列表中并加入标签——————————————————————————————————————#
def loadData(a1,b1,a2,b2):
    class1 = [1] * 500         #前500个为正样本
    class2 = [-1] * 500        #后500个为负样本
    dataMat = []               #数据列表，1000个点
    for i in i_set:
        dataMat.append([float(a1[i]), float(b1[i])])
    for i in i_set:
        dataMat.append([float(a2[i]), float(b2[i])])
    labelMat = []              #标签列表，1000个标签
    for i in i_set:
        labelMat.append([float(class1[i])])
    for i in i_set:
        labelMat.append([float(class2[i])])
    return dataMat,labelMat

def select_random(i, m):         #随机输出一个0到m之间的与i不同的整数
    j = i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

def limit(a, L, H):              #将a限制在L，H之间
    if a > H:
        a = H
    if a < L:
        a = L
    return a

#——————————————————————————————————————————支持向量机————————————————————————————————————————————#
def svm(data_mat, class_label, C, thres, max_iter):
    # 循环外的初始化工作
    data_mat = np.mat(data_mat); label_mat = np.mat(class_label)
    b = 0
    m,n = np.shape(data_mat)
    alphas = np.zeros((m,1))
    iter = 0                     #迭代次数
    while iter < max_iter:
        # 内循环的初始化工作
        alpha_pairs_changed = 0          #用来检验alpha是否改变
        for  i in range(m):
            W = np.dot(np.multiply(alphas, label_mat).T, data_mat)   #W
            f_xi = float(np.dot(W, data_mat[i,:].T)) + b             #fxi=WXi+b
            Ei = f_xi - float(label_mat[i])                          #Ei=fxi-yi

            if((label_mat[i]*Ei < -thres) and  (alphas[i] < C)) or \
            ((label_mat[i]*Ei > thres) and (alphas[i] > 0)):           #判断特征点是不是靠后或者靠前
                j = select_random(i, m)                                #随机选出一个alphaj
                f_xj = float(np.dot(W, data_mat[j,:].T)) + b           #fxj=WXj+b
                Ej = f_xj - float(label_mat[j])                        #Ej=fxj-yj
                alpha_iold = alphas[i].copy()                          #记下此时的alphai，alphaj，之后用于比较其变化情况
                alpha_jold = alphas[j].copy()

                if (label_mat[i] != label_mat[j]):                    #计算alphaj的取值范围L,H
                    L = max(0, alphas[j] - alphas[i])  #
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if H == L :continue

                eta = 2.0 * data_mat[i,:]*data_mat[j,:].T - data_mat[i,:]*data_mat[i,:].T - \
                data_mat[j,:]*data_mat[j,:].T
                if eta >= 0: continue
                alphas[j] = (alphas[j] - label_mat[j]*(Ei - Ej))/eta         #更新alphaj
                alphas[j] = limit(alphas[j], L, H)                           #更新alphaj
                if (abs(alphas[j] - alpha_jold) < 0.00001):
                    continue
                alphas[i] = alphas[i] + label_mat[j]*label_mat[i]*(alpha_jold - alphas[j])   #更新alphai

                # 更新b
                b1 = b - Ei + label_mat[i]*(alpha_iold - alphas[i])*np.dot(data_mat[i,:], data_mat[i,:].T) +\
                label_mat[j]*(alpha_jold - alphas[j])*np.dot(data_mat[i,:], data_mat[j,:].T)
                b2 = b - Ej + label_mat[i]*(alpha_iold - alphas[i])*np.dot(data_mat[i,:], data_mat[j,:].T) +\
                label_mat[j]*(alpha_jold - alphas[j])*np.dot(data_mat[j,:], data_mat[j,:].T)
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alpha_pairs_changed += 1
        print("第",iter,"次迭代","第",i,"次循环")
        #判断迭代是否结束
        #iter += 1
        if (alpha_pairs_changed == 0): iter += 1
        else: iter = iter
    return b, alphas

#————————————————————————————————————————分割结果显示————————————————————————————————————#
def resultshow(dataMat,b1,alphas1):
    dataMat = np.mat(dataMat)

    support_x = []  # 支持向量
    support_y = []

    for i in range(1000):
        if alphas1[i] > 0.0:  # alpha不为0对应的点即为支持向量
            support_x.append(dataMat[i, 0])
            support_y.append(dataMat[i, 1])

    w_best = np.dot(np.multiply(alphas1, labelMat).T, dataMat)
    lin_x = np.linspace(-13, 13)
    lin_y = (-float(b1) - w_best[0, 0] * lin_x) / w_best[0, 1]

    plt.figure(figsize=(10, 10), dpi=125)
    plt.plot(x1, y1, 'bo', label="class=1")  # 绘制上半圆环的点
    plt.plot(x2, y2, 'go', label="class=-1")  # 绘制下半圆环的点
    plt.plot(support_x, support_y, 'rv', label="support_v")  #绘制支持向量
    plt.plot(lin_x, lin_y, color="black")    #绘制分割线

    plt.xlim(-13, 13)  # 坐标轴范围
    plt.ylim(-13, 13)
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
    return w_best

#————————————————————————————————————测试样本测试结果——————————————————————————————————#
def testresult(data,label,WT,b):
    data = np.mat(data);label = np.mat(label)
    rightnum=0
    m, n = np.shape(data)
    for i in range(m):
        fx = float(np.dot(WT, data[i,:].T)) + b
        if i<=499:
            yi=1
        else:
            yi=-1
        if fx>=0 and yi==1:
            print("分割线对第",i,"个测试样本",data[i,:],"的分类结果为1，实际类别标签为1，分类正确！")
            rightnum+=1
        elif fx<0 and yi==-1:
            print("分割线对第", i, "个测试样本", data[i, :], "的分类结果为-1，实际类别标签为-1，分类正确！")
            rightnum+=1
        else:
            print("分割线对第", i, "个测试样本", data[i, :], "的分类错误！")
            rightnum=rightnum

    result=float(rightnum/m)
    print("分割线对测试样本分类结果准确率为",result*100,"%.",m)


#————————————————————————————————————————主函数——————————————————————————————————————#
if __name__=='__main__':
    x1, y1, x2, y2 = trainproduce(0)  # 生成训练集x1,y1/x2,y2
    dataMat, labelMat = loadData(x1, y1, x2, y2)  # 数据列表，标签列表
    b, alphas = svm(dataMat, labelMat, 0.6, 0.001, 10)  #训练b, alphas
    #print(b, alphas)
    WT=resultshow(dataMat,b,alphas)  #显示分割结果

    x11,y11,x22,y22=testproduce(0)          #生成测试集x11,y11/x22,y22
    tdataMat, tlabelMat = loadData(x11, y11, x22, y22)
    testresult(tdataMat,tlabelMat,WT,b)  #测试集测试结果