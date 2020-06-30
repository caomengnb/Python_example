import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import Spinbox
from tkinter import messagebox as mBox
import os
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn import metrics
from Tools import readbunchobj
from yuchuli11 import *

from tfidf11 import *

import pickle
from sklearn.datasets.base import Bunch
from Tools import readfile


###################打开测试集路径##############################
def testpath():
    os.system('explorer.exe /n, test')

'''''
def corpus2Bunch1(wordbag_path, seg_path):

    l1 = name1.get()
    l2 = name2.get()
    l3 = name3.get()
    l4 = name4.get()
    l5 = name5.get()
    l6 = name6.get()
    l7 = name7.get()
    l8 = name8.get()
    print(l1, l2, l3, l4, l5, l6, l7, l8)
    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)

    # 获取每个目录下所有的文件
    for mydir in catelist:
        if mydir==l1 or mydir==l2 or mydir==l3 or mydir==l4 or mydir==l5 or mydir==l6 or mydir==l7 or mydir==l8 :
            class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for file_path in file_list:  # 遍历类别目录下文件
                fullname = class_path +file_path  # 拼出文件名全路class_path + ############################
                bunch.label.append(mydir)
                bunch.filenames.append(fullname)
                bunch.contents.append(readfile(fullname))  # 读取文件内容

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("分词结果向量化结束！")
'''
def corpus2Bunch(wordbag_path, seg_path):
    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)

    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = class_path +file_path  # 拼出文件名全路class_path + ############################
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname))  # 读取文件内容

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("分词结果向量化结束！")


def xianglianghua():
    wordbag_path = "trainxl/trainxl.dat"  # Bunch存储路径
    seg_path = "trainfc/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)
    wordbag_path = "testxl/testxl.dat"  # Bunch存储路径
    seg_path = "testfc/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)
# Create instance
win = tk.Tk()
#win.geometry("1000x600")
# Add a title
win.title("中文文本分类1")

# Disable resizing the GUI
win.resizable(100, 100)

# Tab Control introduced here --------------------------------------
tabControl = ttk.Notebook(win)  # Create Tab Control

tab1 = ttk.Frame(tabControl)  # Create a tab
tabControl.add(tab1, text='第一页')  # Add the tab



tabControl.pack(expand=1, fill="both")  # Pack to make visible
# ~ Tab Control introduced here -----------------------------------------



# ---------------Tab1控件介绍------------------#
# We are creating a container tab3 to hold all other widgets
monty = ttk.LabelFrame(tab1, text='')
monty.grid(column=0, row=0, padx=100, pady=20)

# Changing our Label
#ttk.Label(monty, text="将测试集拖到该目录:").grid(column=0, row=0, sticky='W')
# Adding a Button
action = ttk.Button(monty, text="打开测试集目录", width=12, command=testpath)
action.grid(column=0, row=0, rowspan=2, ipady=9)



def _spin():
    #value = spin.get()
    # print(value)
    fenci()
    xianglianghua()
    tezhen()
    # 导入训练集
    trainpath = "trainxl/traintfdif.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "testxl/testtfidf.dat"
    test_set = readbunchobj(testpath)

    # 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
    clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

    # 预测分类结果
    predicted = clf.predict(test_set.tdm)

    print("分类结束!!!")
    n = 0
    s1=''
    text1=''
    s2=''
    s3=''
    s4=''
    s5=''
    s6=''
    s7=''
    s8=''
    text2 = ''
    text3 = ''
    text4 = ''
    text5 = ''
    text6 = ''
    text7 = ''
    text8 = ''
    c1=''
    c2=''
    c3=''
    c4=''
    c5=''
    c6=''
    c7=''
    c8=''
    for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
        if s1=='' or expct_cate ==s1:

                s1 = expct_cate
                text1=text1+file_name+'   '
        elif s2=='' or expct_cate==s2:

                s2 = expct_cate
                text2 = text2 + file_name + '   '
        elif s3 == '' or expct_cate == s3:

                s3 = expct_cate
                text3 = text3 + file_name + '   '
        elif s4== '' or expct_cate == s4:

                s4 = expct_cate
                text4 = text4 + file_name + '   '
        elif s5 == '' or expct_cate == s5:

                s5 = expct_cate
                text5 = text5 + file_name + '   '
        elif s6 == '' or expct_cate == s6:

                s6 = expct_cate
                text6 = text6 + file_name + '   '
        elif s7 == '' or expct_cate == s7:

                s7 = expct_cate
                text7 = text7 + file_name + '   '
        elif s8 == '' or expct_cate == s8:

                s8 = expct_cate
                text8 = text8 + file_name + '   '

    scr.insert(tk.INSERT, '1.'+ s1 + '：\n')
    scr.insert(tk.INSERT, text1 + '\n\n\n')
    scr.insert(tk.INSERT, '2.' + s2 + '：\n')
    scr.insert(tk.INSERT, text2 + '\n\n\n')
    scr.insert(tk.INSERT, '3.' + s3 + '：\n')
    scr.insert(tk.INSERT, text3 + '\n\n\n')
    scr.insert(tk.INSERT, '4.' + s4 + '：\n')
    scr.insert(tk.INSERT, text4 + '\n\n\n')
    scr.insert(tk.INSERT, '5.' + s5 + '：\n')
    scr.insert(tk.INSERT, text5 + '\n\n\n')
    scr.insert(tk.INSERT, '6.' + s6 + '：\n')
    scr.insert(tk.INSERT, text6 + '\n\n\n')
    scr.insert(tk.INSERT, '7.' + s7 + '：\n')
    scr.insert(tk.INSERT, text7 + '\n\n\n')
    scr.insert(tk.INSERT, '8.' + s8 + '：\n')
    scr.insert(tk.INSERT, text8 + '\n\n\n')

    result = ''
    result = '1.' + s1 + ':' + '\n' + text1 + '\n\n\n' + '2.' + s2 + ':' + '\n' + text2 + '\n\n\n' + '3.' + s3 + ':' + '\n' + text3 + '\n\n\n' + '4.' + s4 + ':' + '\n' + text4 + '\n\n\n' + '5.' + s5 + ':' + '\n' + text5 + '\n\n\n' + '6.' + s6 + ':' + '\n' + text6 + '\n\n\n' + '7.' + s7 + ':' + '\n' + text7 + '\n\n\n' + '8.' + s8 + ':' + '\n' + text8 + '\n\n\n'
    savefile('./result/' + 'result.txt', ''.join(result).encode('utf-8'))
    '''
            if expct_cate != n:
                scr.insert(tk.INSERT, '\n'+expct_cate + '\n')
            else:
                #print("         ", file_name)
                scr.insert(tk.INSERT, file_name + '  ')
            n = expct_cate
'''
# Adding a Button
action = ttk.Button(monty, text="运行", width=10, command=_spin)
action.grid(column=2, row=0, rowspan=2, ipady=7)

# Using a scrolled Text control
scrolW = 100;
scrolH = 30
scr = scrolledtext.ScrolledText(monty, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=0, row=3, sticky='WE', columnspan=3)

# 一次性控制各控件之间的距离
for child in monty.winfo_children():
    child.grid_configure(padx=3, pady=1)
# 单独控制个别控件之间的距离
action.grid(column=2, row=1, rowspan=2, padx=6)
# ---------------Tab1控件介绍------------------#
win.mainloop()

