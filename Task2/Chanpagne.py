# -*- coding: utf-8 -*-
from numpy import *
from pylab import * 

def lwlm(testX, X, Y, tau):
	X = mat(X).T
	Y = mat(Y).T
	m = shape(X)[0]
	weights = mat(eye((m)))
	for j in range(m):
		d = testX - X[j,0]
		weights[j,j] = exp(d * d / (-2.0 * tau ** 2))
	theta1 = (X.T * weights * X).I * X.T * weights * Y
	return theta1

dataFileX = open("q2x.dat", 'r')
dataFileY = open("q2y.dat", 'r')
X = [(float(line)) for line in dataFileX]
Y = [(float(line)) for line in dataFileY]
scatter(X,Y,)
n = 100
x = sort(X)
Tau = [0.03, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
for j in range(8):
	y = [x[i] * float(lwlm(x[i], X, Y, Tau[j])) for i in range(n)]
	subplot(420 + j + 1)
	scatter(X, Y, c = 'y', s = 1, marker = 'o')
	plot(x, y, 'g', linewidth = 1 , label = 'τ =' + str(Tau[j]))
	legend()
show()