# -*- coding: utf-8 -*-
from numpy import *
from pylab import * 

dataFileX = open("q2x.dat", 'r')
dataFileY = open("q2y.dat", 'r')
X = [(float(line)) for line in dataFileX]
Y = [(float(line)) for line in dataFileY]
scatter(X,Y)
theta1 = sum(Y) / sum(X)
theta0 = Y[49] - theta1 * X[49]
alpha = 0.00001
n = 100
time = 0
while True:
	time += 1
	temp1 = 0
	temp0 = 0
	for i in range(0, 99):
		temp1 += (theta0 + theta1 * X[i] - Y[i]) * X[i]
		temp0 += (theta0 + theta1 * X[i] - Y[i])
	old_theta0 = theta0
	old_theta1 = theta1
	theta0 -= alpha * temp0
	theta1 -= alpha * temp1
	x = np.linspace(min(X), max(X), 256, endpoint = True)
	y = theta1 * x + theta0
	plot(x, y)
	if time > 1000:
		break
show()