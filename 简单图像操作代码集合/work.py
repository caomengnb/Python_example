import cmath
import numpy as np
import cv2
import matplotlib.pyplot  as plt
import random
import time
import datetime
import glob
from imutils.object_detection import non_max_suppression

pai = 3.1415926
img = cv2.imread("1.jpg", 1)
height, width, mode = img.shape
print('图像尺寸:',height,width,mode)

#------------------------------彩色图-------------------------------------#
def color():
    cv2.imshow("color_img", img);
    #cv2.waitKey(0);

#------------------------------灰度图-------------------------------------#
dst = np.zeros((height,width,1), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        gray = 0.11*int(b) + 0.59*int(g) + 0.3*int(r) # 为了防止uint8格式数据在运算时发生越界，将其定义为int类型
        dst[i, j] = np.uint8(gray)

def gray():
    cv2.imshow("gray_img", dst)

#------------------------------灰度直方图-------------------------------------#
def plotgray():
    plt.title("grayplot")
    plt.hist(dst.ravel(),256,[0,256])
    plt.show()

#------------------------------灰度图均衡化------------------------------------#
#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst1 = cv2.equalizeHist(dst)
def equalizing():
   cv2.imshow("dst", dst1)

#------------------------------均衡化直方图-------------------------------------#
def plotgray1():
    plt.title("grayplot1")
    plt.hist(dst1.ravel(),256,[0,256])
    plt.show()

#------------------------------梯度锐化-------------------------------------#
tidu = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def tiduhua():
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            aaa = abs(int(dst[i + 1, j]) - int(dst[i, j])) + abs(int(dst[i, j + 1]) - int(dst[i, j]))
            tidu[i, j] = np.uint8(aaa)
    cv2.imshow("tidu", tidu)

#-------------------------------------Laplace增强算子法------------------------------#
laplace = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def LP():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            bbb = 5 * int(dst[i, j]) - int(dst[i + 1, j]) - int(dst[i - 1, j]) - int(dst[i, j + 1]) - int(dst[i, j - 1])
            laplace[i, j] = np.uint8(bbb)
    cv2.imshow("Laplace", laplace)

#-------------------------------------Roberts边缘检测算子------------------------------#
roberts = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def RB():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            ccc = (abs(int(dst[i, j]) - int(dst[i + 1, j + 1])) + abs(int(dst[i + 1, j]) - int(dst[i, j + 1])))
            '''
            if ccc > 125:
                ddd = 0
            else:
                ddd = 255
           '''
            roberts[i, j] = np.uint8(ccc)
    cv2.imshow("Roberts", roberts)

#-------------------------------------Sobel边缘检测算子------------------------------#
sobelx = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
sobely = np.zeros((height,width,mode), np.uint8)
sobel = np.zeros((height,width,mode), np.uint8)
def SB():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            eeex = abs(int(dst[i + 1, j - 1]) + 2 * int(dst[i + 1, j]) + int(dst[i + 1, j + 1]) - int(
                dst[i - 1, j - 1]) - 2 * int(dst[i - 1, j]) - int(dst[i - 1, j + 1]))
            eeey = abs(int(dst[i - 1, j + 1]) + 2 * int(dst[i, j + 1]) + int(dst[i + 1, j + 1]) - int(
                dst[i - 1, j - 1]) - 2 * int(dst[i, j - 1]) - int(dst[i + 1, j - 1]))
            sobelx[i, j] = np.uint8(eeex)
            sobely[i, j] = np.uint8(eeey)
            sobel[i, j] = np.uint8(eeex + eeey)
    cv2.imshow("SobelX", sobelx)
    cv2.imshow("SobelY", sobely)
    cv2.imshow("Sobel", sobel)

#-------------------------------------Laplace边缘检测算子------------------------------#
Laplace = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def LPL():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            fff = abs(
                int(dst[i + 1, j]) + int(dst[i - 1, j]) + int(dst[i, j + 1]) + int(dst[i, j - 1]) - 4 * int(dst[i, j]))
            Laplace[i, j] = np.uint8(fff)
    cv2.imshow("LaplaceL", Laplace)

#-------------------------------------Prewitt边缘检测算子------------------------------#
Prewittx = np.zeros((height,width,mode), np.uint8)
Prewitty = np.zeros((height,width,mode), np.uint8)
Prewitt = np.zeros((height,width,mode), np.uint8)
def PW():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gggx = abs(
                int(dst[i - 1, j - 1]) + int(dst[i - 1, j]) + int(dst[i - 1, j + 1]) - int(dst[i + 1, j - 1]) - int(
                    dst[i + 1, j]) - int(dst[i + 1, j + 1]))
            gggy = abs(
                int(dst[i - 1, j + 1]) + int(dst[i, j + 1]) + int(dst[i + 1, j + 1]) - int(dst[i - 1, j - 1]) - int(
                    dst[i, j - 1]) - int(dst[i + 1, j - 1]))
            if gggx > gggy:
                ggg = gggx
            else:
                ggg = gggy
            Prewittx[i, j] = np.uint8(gggx)
            Prewitty[i, j] = np.uint8(gggy)
            Prewitt[i, j] = np.uint8(ggg)
    cv2.imshow("PrewittX", Prewittx)
    cv2.imshow("PrewittY", Prewitty)
    cv2.imshow("Prewitt", Prewitt)

#-------------------------------------Krisch边缘检测算子------------------------------#
#-------------------------------------[-5 -5 -5]-----------------------------#
#-------------------------------------[ 3  0  3]-----------------------------#
#-------------------------------------[ 3  3  3]-----------------------------#
Krisch = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def KR():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            hhh = 3 * (int(dst[i + 1, j - 1]) + int(dst[i + 1, j]) + int(dst[i + 1, j + 1]) + int(dst[i, j - 1]) + int(
                dst[i, j + 1])) - 5 * (int(dst[i - 1, j - 1]) + int(dst[i - 1, j]) + int(dst[i - 1, j + 1]))
            Krisch[i, j] = np.uint8(hhh / 30)
    cv2.imshow("Krisch", Krisch)

#-------------------------------------Canny边缘检测算子------------------------------#
Canny = np.zeros((height,width,mode), np.uint8)
def CAN():
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            iiix = abs(int(dst[i + 1, j]) - int(dst[i - 1, j]))
            iiiy = abs(int(dst[i, j + 1]) - int(dst[i, j - 1]))
            iii = cmath.sqrt((iiix * iiix) + (iiiy * iiiy))
            if iiiy == 0 and iiix >= 0:
                theta1 = pai / 2
            elif iiiy == 0 and iiix <= 0:
                theta1 = -pai / 2
            else:
                theta1 = cmath.atan(iiix / iiiy)
                theta = theta1.real

            if (theta > pai / 8 and theta <= 3 * pai / 8) or (
                    theta > -7 * pai / 8 and theta <= -5 * pai / 8):  # 45度或-135度
                iii1x = abs(int(dst[i + 2, j + 1]) - int(dst[i, j + 1]))
                iii1y = abs(int(dst[i + 1, j + 2]) - int(dst[i + 1, j]))
                iii1 = cmath.sqrt((iii1x * iii1x) + (iii1y * iii1y))
                iii2x = abs(int(dst[i, j - 1]) - int(dst[i - 2, j - 1]))
                iii2y = abs(int(dst[i - 1, j]) - int(dst[i - 1, j - 2]))
                iii2 = cmath.sqrt((iii2x * iii2x) + (iii2y * iii2y))

            elif (theta > 3 * pai / 8 and theta <= 5 * pai / 8) or (
                    theta > -5 * pai / 8 and theta <= -3 * pai / 8):  # 90度或-90度
                iii1x = abs(int(dst[i, j]) - int(dst[i - 2, j]))
                iii1y = abs(int(dst[i - 1, j + 1]) - int(dst[i - 1, j - 1]))
                iii1 = cmath.sqrt((iii1x * iii1x) + (iii1y * iii1y))
                iii2x = abs(int(dst[i + 2, j]) - int(dst[i, j]))
                iii2y = abs(int(dst[i + 1, j + 1]) - int(dst[i + 1, j - 1]))
                iii2 = cmath.sqrt((iii2x * iii2x) + (iii2y * iii2y))

            elif (theta > 5 * pai / 8 and theta <= 7 * pai / 8) or (
                    theta > -3 * pai / 8 and theta <= -pai / 8):  # 135度或-45度
                iii1x = abs(int(dst[i, j + 1]) - int(dst[i - 2, j + 1]))
                iii1y = abs(int(dst[i - 1, j + 2]) - int(dst[i - 1, j]))
                iii1 = cmath.sqrt((iii1x * iii1x) + (iii1y * iii1y))
                iii2x = abs(int(dst[i + 2, j - 1]) - int(dst[i, j - 1]))
                iii2y = abs(int(dst[i + 1, j]) - int(dst[i + 1, j - 2]))
                iii2 = cmath.sqrt((iii2x * iii2x) + (iii2y * iii2y))

            else:  # 0度或180度
                iii1x = abs(int(dst[i + 1, j + 1]) - int(dst[i - 1, j + 1]))
                iii1y = abs(int(dst[i, j + 2]) - int(dst[i, j]))
                iii1 = cmath.sqrt((iii1x * iii1x) + (iii1y * iii1y))
                iii2x = abs(int(dst[i + 1, j - 1]) - int(dst[i - 1, j - 1]))
                iii2y = abs(int(dst[i, j]) - int(dst[i, j - 2]))
                iii2 = cmath.sqrt((iii2x * iii2x) + (iii2y * iii2y))

            DD = (iii + iii1 + iii2) / 3
            xigma = ((iii - DD) * (iii - DD) + (iii1 - DD) * (iii1 - DD) + (iii2 - DD) * (iii2 - DD)) / 3
            Th1 = DD + xigma
            Th = Th1.real

            if iii.real < iii1.real or iii.real < iii2.real or iii.real < 0.4 * Th:
                out = 0
            else:
                out = iii.real * 5

            Canny[i, j] = np.uint8(out)
    cv2.imshow("Canny",Canny)

#-------------------------------------均值滤波-------------------------------#
#img1 = cv2.imread("jiaoyan.jpg", 1)
#height1, width1, mode1 = img1.shape

#kkk = [0,0,0]
def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros((height,width,1), np.uint8)
    thres = 1 - prob
    for i in range(0, height):
        for j in range(0, width):
            rdn = random.random()
            if rdn < prob:
                output[i,j] = 0
            elif rdn > thres:
                output[i,j] = 255
            else:
                output[i,j] = image[i,j]
    #cv2.imshow("jyzaosheng", output)
    return output

def gasuss_noise(image, mean, var):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

img_spnoise = sp_noise(dst,0.1)
img_gaussnoise = gasuss_noise(dst,0,0.01)
#junzhi = np.zeros((height,width,1), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def ZU1(img1):
    junzhi = np.zeros(img1.shape, np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # kkk1 = [dst[i - 1, j - 1], dst[i - 1, j], dst[i - 1, j + 1], dst[i, j - 1], dst[i, j], dst[i, j + 1], dst[i + 1, j - 1] + dst[i + 1, j] + dst[i + 1, j + 1]]
            kkk1 = [img1[i - 1, j - 1][0], img1[i - 1, j][0], img1[i - 1, j + 1][0], img1[i, j - 1][0], img1[i, j][0],
                    img1[i, j + 1][0], img1[i + 1, j - 1][0], img1[i + 1, j][0], img1[i + 1, j + 1][0]]
            kkk = np.mean(kkk1)
            junzhi[i, j] = np.uint8(kkk)
    return junzhi
def ZU():
    junz = ZU1(img_spnoise)
    junz1 = ZU1(img_gaussnoise)
    cv2.imshow("jyzaosheng", img_spnoise)
    cv2.imshow("MeanLB", junz)
    cv2.imshow("gszaosheng", img_gaussnoise)
    cv2.imshow("MeanLB1", junz1)


#-------------------------------------中值滤波-----------------------------#
#zhong = np.zeros((height,width,1), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
#jjj = [0,0,0]
def ZH1(img1):
    zhong = np.zeros(img1.shape, np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            jjj1 = [int(img1[i - 1, j - 1][0]), int(img1[i - 1, j][0]), int(img1[i - 1, j + 1][0]),
                    int(img1[i, j - 1][0]), int(img1[i, j][0]),
                    int(img1[i, j + 1][0]), int(img1[i + 1, j - 1][0]), int(img1[i + 1, j][0]),
                    int(img1[i + 1, j + 1][0])]
            jjj = np.median(jjj1)
            zhong[i, j] = np.uint8(jjj)
    return zhong
def ZH():
    median = ZH1(img_spnoise)
    median1 = ZH1(img_gaussnoise)
    cv2.imshow("jyzaosheng", img_spnoise)
    cv2.imshow("MedianLB", median)
    cv2.imshow("gszaosheng", img_gaussnoise)
    cv2.imshow("MedianLB1", median1)


#-------------------------------------图像滤波先腐蚀后膨胀-----------------------------#
src = cv2.imread('3.jpg', cv2.IMREAD_UNCHANGED)
kernel = np.ones((13,13), np.uint8)    #卷积核
def imLB():
    erosion = cv2.erode(src, kernel)  # 图像腐蚀处理
    erosion1 = cv2.dilate(erosion, kernel)  # 图像膨胀处理
    cv2.imshow("src", src)
    cv2.imshow("result", erosion1)

#-------------------------------------高斯滤波-----------------------------#
#guassian = np.zeros((height,width,1), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
#lll = [0,0,0]
def gasusslb(out):
    guassian = np.zeros(out.shape, np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lll = (int(out[i - 1, j - 1][0]) + int(out[i - 1, j + 1][0]) + int(out[i + 1, j + 1][0]) + int(
                out[i + 1, j - 1][0]) + 2 * (int(out[i - 1, j][0]) + int(out[i, j + 1][0]) + int(out[i, j - 1][0]) + int(out[i + 1, j][0])) + 4 * int(out[i, j][0])) / 16
            guassian[i, j] = np.uint8(lll)
    return guassian

    #return out
def GU():
    gauss = gasusslb(img_spnoise)
    gauss1 =gasusslb(img_gaussnoise)
    cv2.imshow("jyzaosheng", img_spnoise)
    cv2.imshow("gauss", gauss)
    cv2.imshow("gszaosheng", img_gaussnoise)
    cv2.imshow("guass1", gauss1)



#-------------------------------------图像几何变换------------------------------#
def fangshe():
    pts1 = np.float32([[0, 0], [height-1, 0], [0, width-1]])
    pts2 = np.float32([[height*0, width*0.33], [height*0.85, width*0.25], [height*0.15, width*0.7]])
    M = cv2.getAffineTransform(pts1, pts2)
    # 第三个参数：变换后的图像大小
    res = cv2.warpAffine(img, M, (height, width))
    cv2.imshow("fangshe",res)

#-------------------------------------图像透视变换------------------------------#
def toushi():
    pts1 = np.float32([[0, 0], [height-1, 0], [0, width-1], [height-1, width-1]])
    pts2 = np.float32([[height*0.05, width*0.33], [height*0.9, width*0.25], [height*0.2, width*0.7], [height*0.8, width*0.9]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # 第三个参数：变换后的图像大小
    res = cv2.warpPerspective(img,M,(height,width))
    cv2.imshow("toushi",res)

#-------------------------------------固定阈值分割------------------------------#
guding = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def GD():
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if int(dst[i, j]) >= 125:
                mmm = 255
            else:
                mmm = 0
            guding[i, j] = np.uint8(mmm)
    cv2.imshow("guding", guding)

#-------------------------------------OTSU阈值分割------------------------------#
OSTU = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
All=height*width

def OST():
    W0 = 0;W1 = 0;mu = 0;mu0 = 0;mu1 = 0;g = 0; g1 = 0;TT = 0
    for T in range(10, 235):
        N1 = 0;N0 = 0;sum1 = 0;sum0 = 0;
        for i in range(0, height):
            for j in range(0, height):
                if int(dst[i, j]) >= T:
                    N1 = N1 + 1
                    sum1 = int(sum1) + int(dst[i, j])
                else:
                    N0 = N0 + 1
                    sum0 = int(sum0) + int(dst[i, j])

        W0 = N0/All
        W1 = N1/All
        if N0 == 0 or N1 == 0:
            g1 = 0
        else:
            mu0 = int(sum0 / N0)
            mu1 = int(sum1 / N1)
            mu = W0*mu0+W1*mu1
            g1 =  W0*W1*(mu0-mu1)*(mu0-mu1)
        if g1 >= g:
            g = g1
            TT = T
        else:
            g = g
            TT = TT

    print('T=',TT)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if int(dst[i, j]) >= TT:
                mmm = 255
            else:
                mmm = 0
            OSTU[i, j] = np.uint8(mmm)
    cv2.imshow("OSTU", OSTU)

#-------------------------------------Kittle阈值分割------------------------------#
kittle = np.zeros((height,width,mode), np.uint8) # 新建一个数组，规格与原图片相同，数值为0-255之间的整数
def KT():
    sumu = 0 ;sumg = 0
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            grad1 = abs(int(dst[i + 1, j]) - int(dst[i - 1, j]))
            grad2 = abs(int(dst[i, j + 1]) - int(dst[i, j - 1]))
            if grad1 >= grad2:
                grad = grad1
            else:
                grad = grad2
            uij = grad * int(dst[i, j])
            sumu = sumu + uij
            sumg = sumg + grad
            #print(sumg,sumu)
    T = sumu/sumg
    print('T=', T)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if int(dst[i, j]) >= T:
                mmm = 255
            else:
                mmm = 0
            kittle[i, j] = np.uint8(mmm)
    cv2.imshow("Kittle", kittle)

#-------------------------------------混合高斯背景建模-------------------------------------#
def Gauss():
    colour = ((0, 205, 205), (154, 250, 0), (34, 34, 178), (211, 0, 148), (255, 118, 72), (137, 137, 139))  # 定义矩形颜色
    cap = cv2.VideoCapture("pets2001.avi")  # 参数为0是打开摄像头，文件名是打开视频
    fgbg = cv2.createBackgroundSubtractorMOG2()  # 混合高斯背景建模算法
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 设置保存图片格式
    #out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 10.0,(768, 576))  # 分辨率要和原视频对应

    while True:
        ret, frame = cap.read()  # 读取图片
        fgmask = fgbg.apply(frame)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪

        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景
        count = 0

        for cont in contours:
            Area = cv2.contourArea(cont)  # 计算轮廓面积
            if Area < 300:  # 过滤面积小于10的形状
                continue
            count += 1  # 计数加一
            print("{}-prospect:{}".format(count, Area), end="  ")  # 打印出每个前景的面积
            rect = cv2.boundingRect(cont)  # 提取矩形坐标
            print("x:{} y:{}".format(rect[0], rect[1]))  # 打印坐标
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), colour[count % 6],2)  # 原图上绘制矩形
            cv2.rectangle(fgmask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0xff, 0xff, 0xff),2)  # 黑白前景上绘制矩形
            y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
            cv2.putText(frame, str(count), (rect[0], y-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, colour[count % 6], 2)  # 在前景上写上编号

        cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)  # 显示总数
        cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        print("----------------------------")

        cv2.imshow('frame', frame)  # 在原图上标注
        cv2.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景
        #out.write(frame)
        if cv2.waitKey(30) == 27:  # 按esc退出
            break
    # out.release()#释放文件
    #cap.release()
    cv2.destoryAllWindows()  # 关闭所有窗口

def biaoding():
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob("biaoding\*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点

            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (7, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.imshow('img', img)
            cv2.waitKey(50)

    print(len(img_points))
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数

    print("-----------------------------------------------------")
    # 畸变校正
    img = cv2.imread(images[11])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print(newcameramtx)

    print("------------------使用undistort函数-------------------")
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst1 = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult1.jpg', dst1)
    cv2.imshow('src', img)
    cv2.imshow('biaoding1',dst1)
    print("方法一:dst的大小为:", dst1.shape)

    # undistort方法二
    print("-------------------使用重映射的方式-----------------------")
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)  # 获取映射方程
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)      # 重映射
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)  # 重映射后，图像变小了
    x, y, w, h = roi
    dst2 = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult2.jpg', dst2)
    cv2.imshow('biaoding2', dst2)
    print("方法二:dst的大小为:", dst2.shape)  # 图像比方法一的小

    print("-------------------计算反向投影误差-----------------------")
    tot_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        tot_error += error

    mean_error = tot_error / len(obj_points)
    print("total error: ", tot_error)
    print("mean error: ", mean_error)

def MOD():
    img = cv2.imread("1.jpg", 1)
    img2 = img.copy()
    template = cv2.imread("eye.jpg", 1)
    w, h, m = template.shape

    # 6 中匹配效果对比算法
    methods = ['cv2.TM_CCOEFF']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        print(meth)

        plt.subplot(221), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
        plt.title('template Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()

#--------------------------------------------------ORB特征点匹配------------------------------------------------------------#
def ORB():
    """
    orb特征检测和匹配
        两幅图片分别是 乐队的logo 和包含该logo的专辑封面
        利用orb进行检测后进行匹配两幅图片中的logo

    """
    # 按照灰度图像的方式读入两幅图片
    img1 = cv2.imread("001.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("002.jpg", cv2.IMREAD_GRAYSCALE)

    # 创建ORB特征检测器和描述符
    orb = cv2.ORB_create()
    # 对两幅图像检测特征和描述符
    keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2, None)
    """
    keypoint 是一个包含若干点的列表
    descriptor 对应每个点的描述符 是一个列表， 每一项都是检测到的特征的局部图像

    检测的结果是关键点
    计算的结果是描述符

    可以根据监测点的描述符 来比较检测点的相似之处

    """
    # 获得一个暴力匹配器的对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 利用匹配器 匹配两个描述符的相近成都
    maches = bf.match(descriptor1, descriptor2)
    # 按照相近程度 进行排序
    maches = sorted(maches, key=lambda x: x.distance)
    # 画出匹配项
    img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, maches[: 30], img2, flags=2)

    cv2.imshow("matches", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()

#---------------------------------------------------特征分类器----------------------------------------------------#
def hogsvm():
    hog = cv2.HOGDescriptor()
    hog.load('myHogDector1.bin')

    img = cv2.imread('1.png')

    # cv.imshow('src', img)
    # cv.waitKey(0)

    rects, scores = hog.detectMultiScale(img, winStride=(8, 8), padding=(0, 0), scale=1.05)

    sc = [score[0] for score in scores]
    sc = np.array(sc)

    # 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
    for i in range(len(rects)):
        r = rects[i]
        rects[i][2] = r[0] + r[2]
        rects[i][3] = r[1] + r[3]

    pick = []
    # 非极大值移植
    print('rects_len', len(rects))
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    print('pick_len = ', len(pick))

    # 画出矩形框
    for (x, y, xx, yy) in pick:
        cv2.rectangle(img, (x, y), (xx, yy), (0, 0, 255), 2)

    cv2.imshow('a', img)
    cv2.waitKey(0)

#-----------------------------------------------------------CamShift------------------------------------------------------------#
xs, ys, ws, hs = 0, 0, 0, 0  # selection.x selection.y
xo, yo = 0, 0  # origin.x origin.y
selectObject = False
trackObject = 0

def onMouse(event, x, y, flags, prams):
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject

    if selectObject == True:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1

def camShift():
    global trackObject,roi_hist,track_window
    cap = cv2.VideoCapture('1.mp4')
    ret, frame = cap.read()
    cv2.namedWindow('imshow')
    cv2.setMouseCallback('imshow', onMouse)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1300, 750), interpolation=cv2.INTER_CUBIC)
        if trackObject != 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
            if trackObject == -1:
                track_window = (xs, ys, ws, hs)
                maskroi = mask[ys:ys + hs, xs:xs + ws]
                hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            dst &= mask
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)

        if selectObject == True and ws > 0 and hs > 0:
            cv2.imshow('imshow1', frame[ys:ys + hs, xs:xs + ws])
            cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])
        cv2.imshow('imshow', frame)
        if cv2.waitKey(30) == 27:
            break
    cv2.destroyAllWindows()