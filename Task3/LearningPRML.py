# -*- coding:utf-8 -*-
'''
Created on 2017年4月3日

@author: 郑康杰
'''
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.sql.expression import false

X = []
Y = []
theta = []
dimension = 2   #输入数据维度，不可调整
power = 5.0  #边界曲线的次数可以调整
max_itor = 80000   #最大迭代次数
alpha = 0.01  #学习率
epsilon = 0.00000001
INF = 1000000000
PROP = 1.1   #控制图表缩放比例系数
x_max_range = []
x_min_range = []
kind1_x = []
kind1_y = []
kind2_x = []
kind2_y = []
def LoadDataSet():
    file1=open('task3_x.dat')
    dataX=file1.readlines()
    file2=open('task3_y.dat')
    dataY=file2.readlines()
    file1.close()
    file2.close()
    x = []
    x_max = []
    x_min = []
    sum = []
    n = 0
    for i in range(dimension):
        sum.append(0.0)
        x_max.append(-INF)
        x_min.append(INF)
        x_max_range.append(-INF)
        x_min_range.append(INF)
    for eachline in dataX:
        n += 1
        str = eachline.lstrip()
        tmp = []
        for i in range(dimension):
            if(float(str.split()[i])>x_max[i]):
                x_max[i]=float(str.split()[i]);
            if(float(str.split()[i])<x_min[i]):
                x_min[i]=float(str.split()[i]);
            sum[i] += float(str.split()[i])
            #tmp.append(float(str.split()[i]))
    average = [(sum[i]/float(n))for i in range(dimension)]
    for eachline in dataX:
        str = eachline.lstrip()
        tmp = []
        x = []
        for i in range(dimension):
            New_x = (float(str.split()[i])-average[i])/(x_max[i]-x_min[i])
            if(New_x > x_max_range[i]): x_max_range[i] = New_x
            if(x_min_range[i] > New_x): x_min_range[i] = New_x
            x.append(New_x)
        for i in range(int(power)+1):
            for j in range(i+1):
                tmp.append(x[0]**(i-j)*x[1]**j)
        X.append(tmp)
    for eachline in dataY:
        Y.append(float(eachline))
    n = len(X)
    for i in range(int((power+2)*(power+1)/2)):
        theta.append(0.0)

def Sigmoid(x):
    return (1 / (1 + np.exp(-x)))

LoadDataSet()
for i in range(len(Y)):
    if(Y[i]==0.0):
        kind1_x.append(X[i][1])
        kind1_y.append(X[i][2])
    else:
        kind2_x.append(X[i][1])
        kind2_y.append(X[i][2])
plt.scatter(kind1_x,kind1_y,s=20)
plt.scatter(kind2_x,kind2_y,s=20)
x_mat = mat(X)
y_mat = mat(Y).T
theta_mat = mat(theta).T
itor = 0
while True:
    itor += 1
    A = x_mat*theta_mat
    E = Sigmoid(A) - y_mat
    theta_mat = theta_mat - alpha*x_mat.T*E
    if(itor%100 == 0 and itor != 0):
        print("itor:"+str(itor))
    if(itor == max_itor):
        break
X_Scale = []
for i in range(dimension):
    X_Scale.append(np.arange(x_min_range[i]*PROP, x_max_range[i]*PROP, (x_max_range[i]-x_min_range[i])/80.0))
a,b = np.meshgrid(X_Scale[0],X_Scale[1])
f = theta_mat[0, 0] 
for i in range(int(power)+1):
    for j in range(i+1):
        pos = int(float((i+1)*i)/2.0) + j
        f += theta_mat[pos, 0]*a**(i-j)*b**j
f -= theta_mat[0, 0] 
plt.contour(a,b,f,0)
plt.show()