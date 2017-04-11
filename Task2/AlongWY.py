# -*- coding:utf-8 -*-
#局部加权线性回归

import numpy as np
import matplotlib.pyplot as plt

#读入X,Y数据
X = [(float)(x.strip()) for x in open('q2x.dat').readlines()]
Y = [(float)(y.strip()) for y in open('q2y.dat').readlines()]

#对X,Y进行向量化处理
XV = np.matrix([(1,x) for x in X])
YV = np.matrix([(y,) for y in Y])

#向量输出测试
#print(XV)
#print(YV)

#对点计算拟和，给定点x值，波长值，数据集XV,YV，返回theta
def theta(x,XV,YV,K):
    num = len(XV)
    #计算权重矩阵
    W = np.mat(np.eye(num))
    for i in range(num):
        W[i,i] = np.exp((x-XV[i,1])**2/(-2*K*K))
    XTWT=XV.T*W*XV
    return XTWT.I*XV.T*W*YV

#对给定的X序列进行拟合
def tofitter(XL,XV,YV,K,c = 'r',s = 10,lb = '-'):
    YL = []
    for x in XL:
        y = np.matrix([1,x])*theta(x,XV,YV,K)
        YL.append(y[0,0])
    #绘制曲线
    plt.plot(XL,YL,c,s,linestyle='-',label=lb)
	

#设置X,Y轴的上下限
plt.xlim(-5.2,13.0)
plt.ylim(-2.2,2.5)

#散点绘制
plt.scatter(X,Y,c='r', s=10, marker = 'x',label='Traing Data')

#为接下来绘制线条做准备
X.sort()

#tau=0.1
tofitter(X,XV,YV,0.1,c = 'c',s = 10,lb = 'tau=0.1')

#tau=0.3
tofitter(X,XV,YV,0.3,c = 'k',s = 10,lb = 'tau=0.3')

#tau=1.0
tofitter(X,XV,YV,1.0,c = 'm',s = 10,lb = 'tau=1.0')

#tau=3.0
tofitter(X,XV,YV,3.0,c = 'w',s = 10,lb = 'tau=3.0')

#tau=10.0
tofitter(X,XV,YV,10.0,c = 'y',s = 10,lb = 'tau=10.0')

#tau=100.0
tofitter(X,XV,YV,100,c = 'r',s = 10,lb = 'tau=100.0')

#设置标题
plt.title("AlongWY's Graphics")

#显示界面
plt.legend()
plt.show()