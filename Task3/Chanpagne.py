# -*- coding: utf-8 -*-
from numpy import *
from pylab import *

tempX = [[float(a) for a in x.split()] for x in open("task3_x.dat").readlines()]
Y = [float(line) for line in open("task3_y.dat").readlines()]
max1 = -inf
min1 = inf
max2 = -inf
min2 = inf
X = []
for x1, x2 in tempX:
	max1 = max(x1, max1)
	min1 = min(x1, min1)
	max2 = max(x2, max2)
	min2 = min(x2, min2)
	X.append((1, x1, x2, x1**2, x1*x2, x2**2, x1**3, (x1**2)*x2, x1*(x2**2), x2**3,
		x1**4, (x1**3)*x2, (x1**2)*(x2**2), x1*(x2**3), x2**4,x1**5, (x1**4) * x2, (x1**3)*(x2**2), (x1**2)*(x2**3),
		x1*(x2**4), x2**5))
theta = ones((21,1))
m = size(Y)
time = 10000
alpha = 0.01
for i in range(m):
	if Y[i] == 1.0:
		scatter(X[i][1], X[i][2], c = 'r', marker = '+')
	else:
		scatter(X[i][1], X[i][2], c = 'b')
X = mat(X)
Y = mat(Y).T
for i in range(time):
	error = Y - 1.0 / (1.0 + exp(-X * theta))
	theta += alpha * X.T * error
x1 = arange(min1, max1, .01)
x2 = arange(min2, max2, .01)
x1, x2 = meshgrid(x1, x2)
f = theta[0, 0] + theta[1, 0]*x1 + theta[2, 0]*x2 + theta[3, 0]*x1**2 + theta[4, 0]*x1*x2 + theta[5, 0]*x2**2 \
	+ theta[6, 0]*x1**3+theta[7, 0]*(x1**2)*x2 + theta[8, 0]*x1*(x2**2) + theta[9, 0]*x2**3 + theta[10, 0]*x1**4 \
	+ theta[11, 0]*(x1**3)*x2 + theta[12, 0]*(x1**2)*(x2**2) +theta[13, 0]*x1*(x2**3) + theta[14, 0]*x2**4 + \
	+ theta[15, 0]*x1**5 + theta[16, 0]*(x1**4) * x2 + theta[17, 0]*(x1**3)*(x2**2) + theta[18, 0]*(x1**2)*(x2**3) \
	+ theta[19, 0]*x1*(x2**4) + theta[20, 0]*x2**5
contour(x1, x2, f, 0)
show()