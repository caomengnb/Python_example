# 说明
# data文件夹里是MNIST数据集
# work文件夹里

readdata.py用来解析MNIST数据集

gParam.py为全局参数

cnn.py为卷积网络类，包括参数初始化、网络各层算法函数、误差反向传播函数等

train.py训练卷积网络，并将关键参数存储到train_result文件夹

test.py读取前面保存的参数，检测手写数字