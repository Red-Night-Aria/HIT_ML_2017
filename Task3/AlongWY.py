# -*- coding:utf-8 -*-
from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt

#读取文件
X = []
for line in open('task3_x.dat').readlines():
    x1,x2 = map(float,line.strip().split(' '))
    X.append((1,x1,x2,x1*x2,x1**2,x2**2,x1**2*x2,x1*x2**2,x1**3,x2**3))
    
X = mat(X)
Y = mat([(float)(y.strip()) for y in open('task3_y.dat').readlines()]).T

#构造预测函数
def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))
        
#构造评估J函数并求出theta
def gradAscent(dataMatrix,LabelMat):
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 10000
    theta = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * theta)
        error = (LabelMat - h)
        theta = theta + alpha * dataMatrix.T * error
    return theta
    
theta = gradAscent(X,Y)
print(theta)

AX = []
AY = []
BX = []
BY = []
for i in range(len(X)):
    if Y[i,0]==0:
        AX.append(X[i,1])
        AY.append(X[i,2])
    if Y[i,0]==1:
        BX.append(X[i,1])
        BY.append(X[i,2])
        
plt.scatter(AX,AY,color = 'b',marker = 'x')
plt.scatter(BX,BY,color = 'r',marker = '+')

X1 = arange(-4,8,0.1)
X2 = arange(-4,4,0.1)

x1,x2 = meshgrid(X1,X2)
f = theta[0,0] + theta[1,0]*x1 + theta[2,0]*x2 +theta[3,0]*x1*x2 +theta[4,0]*x1*x1+theta[5,0]*x2*x2+theta[6,0]*x1**2*x2+theta[7,0]*x1*x2**2+theta[8,0]*x1**3+theta[9,0]*x2**3

#设置标题
plt.title("AlongWY's Graphics")

plt.contour(x1,x2,f,0)
plt.show()