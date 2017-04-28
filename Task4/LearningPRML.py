# -*- coding:utf-8 -*-
'''
Created on 2017年4月16日

@author: 郑康杰
'''
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
import random as rand

X = []
Y = []
kind1_x = []
kind1_y = []
kind2_x = []
kind2_y = []
N = 10   #隐层神经元数目
max_itor = 100000   #最大迭代次数
alpha = 0.01  #学习率

def Sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def LoadDataSet():
    file=open('watermelon.txt')
    dataSet=file.readlines()
    file.close()
    for eachline in dataSet:
        if(eachline[0]=='编'):
            continue
        str = eachline.split()
        X.append([1.0,float(str[1]),float(str[2])])
        if(str[3] == '是'):
            Y.append(1.0)
            kind1_x.append(float(str[1]))
            kind1_y.append(float(str[2]))
        else:
            Y.append(0.0)
            kind2_x.append(float(str[1]))
            kind2_y.append(float(str[2]))
    plt.scatter(kind1_x,kind1_y,s=50)
    plt.scatter(kind2_x,kind2_y,s=50)

LoadDataSet()
n = len(Y)
X_mat = mat(X).T
Y_mat = mat(Y).T
#W1 = mat(zeros((N+1,3)))
#W2 = mat(zeros((N+1,1)))
WTF = []
for i in range(N):
    WTF.append([rand.random(), rand.random(), rand.random()])
W1 = mat(WTF)
WTF = []
for i in range(N+1):
    WTF.append([rand.random()])
W2 = mat(WTF)
itor = 0
while True:
    itor += 1
    DELTA1 = mat(zeros((N+1,3)))
    DELTA2 = mat(zeros((N+1,1)))
    A1 = mat(X).T
    Z2 = W1 * A1   #W1:(N,3),A1:(3,n)
    A2 = mat(ones((1,n)))
    A2 = np.row_stack((A2,Sigmoid(Z2))) #A2:(N+1,n)
    Z3 = A2.T * W2 # A2.T:(n,N+1),W2:(N+1,1)
    A3 = Sigmoid(Z3) #A3:(n,1)
    delta3 = A3 - Y_mat #delta3:(n,1)
    delta2 = np.multiply(W2 * delta3.T , np.multiply(A2 , (1-A2)))# W2:(N+1,1),delta3:(n,1),W2.T:(1,N+1)
    DELTA1 += delta2 * A1.T #delta2:(N+1,n)  ,A1.T:(n,3) DELTA1:(N+1,3)       
    #print(delta3.shape,A2.T.shape,DELTA2.shape)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    DELTA2 += A2 *delta3 # A2:(N+1,n),delta3:(n,1)
    W1 -= alpha*DELTA1[1:][:] 
    W2 -= alpha*DELTA2  #W2:(N+1,1)
    if(itor % 100 == 0):
        print("itor:"+str(itor))
    if(itor == max_itor):
        break
x1 = np.arange(0, 0.8, .01)
x2 = np.arange(0, 0.6, .01)
x1, x2 = np.meshgrid(x1, x2)
f = W2[0,0]
for i in range(N):
    tmp = W1[i,0]+W1[i,1]*x1+W1[i,2]*x2
    f += W2[i+1,0] * Sigmoid(tmp)
plt.contour(x1, x2, f, 0)
plt.show()

