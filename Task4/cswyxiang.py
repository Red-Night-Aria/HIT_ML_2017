import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import random as rand
#读数据
file = open("watermelon.txt")
date = file.readlines()

X = []
Y = []

for eachline in date:
    x = eachline.split()
    if x[0] == "编号":
        continue
    else:
        X.append((1.0,float(x[1]),float(x[2])))
        Y.append(float(x[3]=='是'))
X = np.mat(X).T
Y = np.mat(Y)

m = shape(Y)[1]
alpha = 0.01
step = 300000

def sigmoid(z):
    return 1/(1+exp(-z))
#随机初始化theta1
theta1 = mat([[rand.random(),rand.random(),rand.random(),rand.random(),rand.random()],[rand.random(),rand.random(),rand.random(),rand.random(),rand.random()],[rand.random(),rand.random(),rand.random(),rand.random(),rand.random()]]) #theta1:3*5
#随机初始化theta2
theta2 = mat([rand.random(),rand.random(),rand.random(),rand.random(),rand.random(),rand.random()]) #theta2:1*6

D1 = mat(zeros([shape(theta1)[0],shape(theta1)[1]])) #3*5
D2 = mat(zeros([shape(theta2)[0],shape(theta2)[1]])) #1*6

for i in range(step):
    #计算隐藏层
    a1 = sigmoid(theta1.T*X) #a1:5*17
    b2 = mat(np.ones((1, m))) #b2:1*17
    X1 = np.row_stack((b2, a1))  #X1:6*17
    #计算输出层
    output = sigmoid(theta2*X1) #output:1*17
    delta2 = output - Y #delta2: 1*17
    delta1 = np.multiply(theta2.T[1:,:]*delta2 , np.multiply(a1,(1-a1))) #delta1:6*17
    #D1 = D1 + X*delta1.T
    #D2 = D2 + delta2*X1.T
    D1 =  X*delta1.T
    D2 =  delta2*X1.T
    #更新theta    
    theta1 = theta1 - alpha*D1
    theta2 = theta2 - alpha*D2

#画散点图
t = 0;
for i in range(shape(Y)[1]):  
    each = Y[0,i]
    if each > 0:
        plt.scatter(X[1,t],X[2,t],color = 'red',marker = "x")
    else:
        plt.scatter(X[1,t],X[2,t],color = 'blue')
    t = t + 1

x1 = np.arange(0, 0.8, .01)
x2 = np.arange(0, 0.6, .01)
x1,x2 = np.meshgrid(x1,x2)

f = theta2[0,0]
for k in range(shape(theta1)[1]):
    f += sigmoid(theta1[0,k]+theta1[1,k]*x1+theta1[2,k]*x2)*theta2[0,k+1]
plt.contour(x1,x2,f,0)
plt.show()
