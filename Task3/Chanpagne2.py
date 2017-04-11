# -*- coding: utf-8 -*-
from numpy import *
from pylab import *
#牛顿法
tempX = [[float(a) for a in x.split()] for x in open("task3_x.dat").readlines()]
Y = [float(line) for line in open("task3_y.dat").readlines()]
X = []
for x1, x2 in tempX:
	X.append((1, x1, x2, x1**2, x1*x2, x2**2, x1**3, (x1**2)*x2, x1*(x2**2), x2**3,
		x1**4, (x1**3)*x2, (x1**2)*(x2**2), x1*(x2**3), x2**4))
theta = zeros((15,1))
m = size(Y)
time = 100
for i in range(m):
	if Y[i] == 1.0:
		scatter(X[i][1], X[i][2], c = 'r', marker = '+')
	else:
		scatter(X[i][1], X[i][2], c = 'b')
X = mat(X)
Y = mat(Y).T
for i in range(time):
	print(i)
	error = 1.0 / (1.0 + exp(-X * theta)) - Y
	error1 = multiply(1.0/(1.0+exp(-X*theta)),1.0 - 1.0 / (1.0 + exp(-X * theta)))
	temp = X
	for k in range(m):
		for j in range(15):
			temp[k, j] *= error1[k, 0]
	theta -= ((X.T * temp )* 1.0 / m).I * (X.T * error) * 1.0 / m
x1 = arange(-4.0, 8.0, .01)
x2 = arange(-4.0, 4.0, .01)
x1, x2 = meshgrid(x1, x2)
f = theta[0, 0] + theta[1, 0]*x1 + theta[2, 0]*x2 + theta[3, 0]*x1**2 + theta[4, 0]*x1*x2 + theta[5, 0]*x2**2 \
	+ theta[6, 0]*x1**3+theta[7, 0]*(x1**2)*x2 + theta[8, 0]*x1*(x2**2) + theta[9, 0]*x2**3 + theta[10, 0]*x1**4 \
	+ theta[11, 0]*(x1**3)*x2 + theta[12, 0]*(x1**2)*(x2**2) +theta[13, 0]*x1*(x2**3) + theta[14, 0]*x2**4
contour(x1, x2, f, 0)
show()