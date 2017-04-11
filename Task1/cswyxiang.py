import numpy as np
import matplotlib.pyplot as plt
from numpy import *

file1 = open("q2x.dat")
dataX = file1.readlines()
file2 = open("q2y.dat")
dataY = file2.readlines()

file1.close()
file2.close()

X = []
Y = []

for eachline in dataX:
	X.append(float(eachline))
for eachline in dataY:
	Y.append(float(eachline))

pointX = np.mat(X)
pointY = np.mat(Y)

plt.scatter(pointX,pointY)

m = size(X)
theta0 = 0
theta1 = 0
alpha = 0.01
Max_step = 500
e = 0.0001

for i in range(Max_step):
	sigma0 = 0
	sigma1 = 0
	for j in range(m):
		sigma0 = sigma0 + theta0 + theta1 * pointX[0,j] - pointY[0,j]
		sigma1 = sigma1 + (theta0 + theta1 * pointX[0,j] - pointY[0,j]) * pointX[0,j]
	theta0 = theta0 - alpha * (1/m) * sigma0
	theta1 = theta1 - alpha * (1/m) * sigma1
	x = np.linspace(-6, 13, 100)
	y = theta0 + theta1 * x
	plt.plot(x,y,'blue')

plt.legend()
plt.show()
