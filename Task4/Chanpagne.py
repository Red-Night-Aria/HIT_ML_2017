# -*- coding: utf-8 -*-
from numpy import *
from pylab import *
import random as rd

def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

X = []
Y = []
for x in open("watermelon.txt").readlines():
	if x[0] == '编':
		continue
	temp = [a for a in x.split()]
	X.append([float(temp[1]), float(temp[2])])
	if temp[3] == '是':
		Y.append(1.0)
	else:
		Y.append(0.0)
m = size(Y)
for i in range(m):
	if Y[i] == 1.0:
		scatter(X[i][0], X[i][1], c = 'r', marker = '+')
	else:
		scatter(X[i][0], X[i][1], c = 'b')
X = mat(X)
Y = mat(Y).T
theta = rd.random()     			   															#输出层阈值
gamma = mat([rd.random(), rd.random(), rd.random()]).T     										#隐层的阈值 3 X 1
v = mat(([rd.random(), rd.random(), rd.random()], [rd.random(), rd.random(), rd.random()]))     #输入到隐层的连接权 2 X 3
w = mat((rd.random(), rd.random(), rd.random())).T		   										#隐层到输出的连接权(3 X 1)
alpha = ones((3, 1))				   															#隐层接收到的输入   3 X 1
#b					  																			#隐层的输出(3 X 1)
#beta			 	 																			#输出层接收到的输入
e = ones((3, 1))
yita = 0.1
time = 10000
for t in range(time):
	for i in range(m):
		alpha = v.T * mat(X[i]).T
		b = sigmoid(alpha - gamma)
		beta = b.T * w
		tempY = sigmoid(beta - theta)
		g = float(tempY * (1 - tempY) * (Y[i,0] - tempY))
		for h in range(3):
			e[h,0] = b[h,0] * (1 - b[h,0]) * w[h,0] * g
		w += yita * g * b
		v += yita * mat(X[i]).T * e.T
		theta += -yita * g
		gamma += -yita * e 
x1 = arange(0, 0.85, .01)
x2 = arange(0, 0.55, .01)
x1, x2 = meshgrid(x1, x2)
f = sigmoid(x1 * v[0,0] + x2 * v[1,0] - gamma[0,0]) * w[0,0] + \
 sigmoid(x1 * v[0,1] + x2 * v[1,1] - gamma[1,0]) * w[1,0] + sigmoid(x1 * v[0,2] + x2 * v[1,2] - gamma[2,0]) * w[2,0] - theta
contour(x1, x2, f, 0)
show()
