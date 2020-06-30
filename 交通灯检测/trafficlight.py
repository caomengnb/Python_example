import cv2
import numpy as np
import cmath
import os

#找到高亮区域，判断面积大小，颜色，绿色看上方两点，红色看下方两点，分别上下延伸
def lightdetect(gray,src):
    contours, heirs = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rec = [0]*4

    for cont in contours:
        Area = cv2.contourArea(cont)  # 计算轮廓面积
        rect = cv2.boundingRect(cont)  # 提取矩形坐标
        if Area < 2 or Area >500 or rect[1]>750*14/29:  # 过滤面积小于10的形状
            continue

        #rect = cv2.boundingRect(cont)  # 提取矩形坐标
        xi = (2 * int(rect[0]) + int(rect[2])) / 2
        yi = (2 * int(rect[1]) + int(rect[3])) / 2
        xi = int(xi)
        yi = int(yi)
        b, g, r = src[yi, xi]

        if g>r:
            yi=yi-rect[3];yj= yi -rect[3]
            b1,g1,r1=src[yi,xi];b2,g2,r2=src[yj,xi]
            if b1<150 and g1<150 and r1<150 and b2<150 and g2<150 and r2<150:
               rec[0]=rect[0]-6
               rec[1]=rect[1]-6-4*rect[3]
               rec[2]=rect[0] + rect[2]+6
               rec[3]=rect[1] + rect[3]+6
               cv2.rectangle(src, (rec[0], rec[1]), (rec[2] , rec[3]), (0, 0, 255),2)
               cv2.putText(src, "green", (rec[0] - 6, rec[1] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        else:
            yi = yi + rect[3];yj= yi +rect[3]
            b1, g1, r1 = src[yi, xi];
            b2, g2, r2 = src[yj, xi]
            if b1 < 150 and g1 < 150 and r1 < 150 and b2 < 150 and g2 < 150 and r2 < 150:
                rec[0] = rect[0] - 6
                rec[1] = rect[1] - 6
                rec[2] = rect[0] + rect[2] + 6
                rec[3] = rect[1] + 5*rect[3] + 6
                cv2.rectangle(src, (rec[0], rec[1]), (rec[2], rec[3]), (0, 0, 255), 2)
                cv2.putText(src, "red", (rec[0]-6, rec[1]-6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)


#输入文件路径
filePath = "1.jpg"
src = cv2.imread(filePath)
img = cv2.resize(src, (1000,750), interpolation = cv2.INTER_CUBIC)
#调整图像大小
src = cv2.resize(src,(1000,750),src,0,0,cv2.INTER_AREA)

src_1 = src.copy()
shape = src.shape
#1、检测红色和绿色区域
for i in range(shape[0]):
    for j in range(shape[1]):
        B,G,R = src_1[i][j]
        if (B > 150 and G > 150 and R > 150) or (B < 50 and G < 50 and R < 50) or (B>G and B>R):
            src_1[i][j] = [0,0,0]

        #B, G, R = hsv[i][j]
        S = abs(int(G) - int(R))
        if S<=100:# and (G<100 or R<100):
            src_1[i][j] = [0, 0, 0]



#对图像进行灰度化
gray = cv2.cvtColor(src_1,cv2.COLOR_RGB2GRAY)

lightdetect(gray,src)

cv2.imshow('src1',img)
cv2.imshow('src_1',src_1)
cv2.imshow('gray',gray)
cv2.imshow('src',src)

cv2.waitKey(0)
cv2.destroyAllWindows()

