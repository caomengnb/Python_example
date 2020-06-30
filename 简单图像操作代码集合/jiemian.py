import tkinter as tk
from tkinter import ttk
import glob
from work import color,gray,plotgray,equalizing,plotgray1,tiduhua,LP,RB,SB,LPL,PW,KR
from work import CAN,ZU,ZH,imLB,GU,GD,OST,fangshe,toushi,KT,Gauss,biaoding,MOD,ORB,hogsvm,camShift


#------------------------生成一个弹窗-------------------------------#
win = tk.Tk()
# 标题
win.title("图像处理")
#------------------------生成一个弹窗-------------------------------#



# -----------------------生成多个Tab--------------------------------#
tabControl = ttk.Notebook(win)  # Create Tab Control

tab1 = ttk.Frame(tabControl)  # Create a tab
tabControl.add(tab1, text='灰度化|均衡|锐化')  # Add the tab

tab2= ttk.Frame(tabControl)  # Create a tab
tabControl.add(tab2, text='边缘检测|滤波')  # Add the tab

tab3= ttk.Frame(tabControl)  # Create a tab
tabControl.add(tab3, text='图像校正|标定|目标检测')  # Add the tab

tabControl.pack(expand=1, fill="both")  # Pack to make visible
# -----------------------生成多个Tab--------------------------------#



# ---------------Tab1控件介绍------------------#
# We are creating a container tab3 to hold all other widgets
monty = ttk.LabelFrame(tab1, text='')
monty.grid(column=0, row=0, padx=200, pady=150)

# Adding a Button
action = ttk.Button(monty, text="彩色图", width=12, command=color)
action.grid(column=1, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="形态学滤波", width=12, command=imLB)
action.grid(column=1, row=3, rowspan=2, ipady=9)

# Adding a Button
action = ttk.Button(monty, text="灰度图", width=12,command=gray)
action.grid(column=2, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="灰度直方图", width=12,command=plotgray)
action.grid(column=2, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="灰度图均衡化", width=12,command=equalizing)
action.grid(column=3, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="均衡化直方图", width=12,command=plotgray1)
action.grid(column=3, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="梯度锐化", width=12,command=tiduhua)
action.grid(column=4, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Laplace", width=12, command=LP)
action.grid(column=4, row=3, rowspan=2, ipady=9)

for child in monty.winfo_children():
    child.grid_configure(padx=3, pady=1)
# 单独控制个别控件之间的距离
action.grid(column=4, row=3, rowspan=3, padx=6)
# ---------------Tab1控件介绍------------------#

# ---------------Tab2控件介绍------------------#
# We are creating a container tab3 to hold all other widgets
monty = ttk.LabelFrame(tab2, text='')
monty.grid(column=0, row=0, padx=200, pady=150)

# Adding a Button
action = ttk.Button(monty, text="Roberts", width=12,command=RB)
action.grid(column=1, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Sobel", width=12,command=SB)
action.grid(column=1, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Laplace1", width=12,command=LPL)
action.grid(column=2, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Prewitt", width=12,command=PW)
action.grid(column=2, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Krisch", width=12,command=KR)
action.grid(column=3, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Canny", width=12,command=CAN)
action.grid(column=3, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="均值滤波", width=12, command=ZU)
action.grid(column=4, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="中值滤波", width=12, command=ZH)
action.grid(column=4, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="高斯滤波", width=12, command=GU)
action.grid(column=5, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="单目相机标定", width=12, command=biaoding)
action.grid(column=5, row=3, rowspan=2, ipady=9)

for child in monty.winfo_children():
    child.grid_configure(padx=3, pady=1)
# 单独控制个别控件之间的距离
action.grid(column=5, row=3, rowspan=3, padx=6)
# ---------------Tab2控件介绍------------------#

# ---------------Tab3控件介绍------------------#
# We are creating a container tab3 to hold all other widgets
monty = ttk.LabelFrame(tab3, text='')
monty.grid(column=0, row=0, padx=200, pady=150)

# Adding a Button
action = ttk.Button(monty, text="仿射变换", width=12, command=fangshe)
action.grid(column=1, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="透视变换", width=12, command=toushi)
action.grid(column=1, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="固定阈值分割", width=12,command=GD)
action.grid(column=2, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="OSTU阈值", width=12,command=OST)
action.grid(column=2, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="Kittle", width=12,command=KT)
action.grid(column=3, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="混合高斯背景", width=12,command=Gauss)
action.grid(column=3, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="模板匹配", width=12, command=MOD)
action.grid(column=4, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="特征点匹配", width=12, command=ORB)
action.grid(column=4, row=3, rowspan=2, ipady=9)

action = ttk.Button(monty, text="特征分类器", width=12, command=hogsvm)
action.grid(column=5, row=1, rowspan=2, ipady=9)

action = ttk.Button(monty, text="CamShift", width=12, command=camShift)
action.grid(column=5, row=3, rowspan=2, ipady=9)

for child in monty.winfo_children():
    child.grid_configure(padx=3, pady=1)
# 单独控制个别控件之间的距离
action.grid(column=5, row=3, rowspan=3, padx=6)
# ---------------Tab3控件介绍------------------#

win.mainloop()