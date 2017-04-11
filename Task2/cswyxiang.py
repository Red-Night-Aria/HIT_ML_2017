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

tau = [0.01,0.03,0.1,0.3,1.0,3.0,10.0,100.0]
color = ['gold','brown','peru','greenyellow','gray','blueviolet','tomato','olive']


for eachline in dataX:
	X.append(float(eachline))
for eachline in dataY:
	Y.append(float(eachline))

m = size(X);

def lwr(x, pointX, pointY, tau):
	w = mat(eye(m))
	for i in range(m):
		w[i,i] = exp((-(x-pointX[0,i])**2)/(2*tau**2))
	V = pointX * w * pointX.T
	theta = V.I * pointX * w * pointY.T
	return theta[0,0]

for j in range(8):

	pointX = np.mat(X)
	pointY = np.mat(Y)

	X1 = []
	Y1 = []

	for x1 in X:
		X1.append(x1)
	X1.sort(key=None, reverse=False)

	for k in range(m):
		Y1.append(lwr(X1[k],pointX,pointY,tau[j])* X1[k])
	plt.plot(X1,Y1,color = color[j], label = 'tau = ' + str(tau[j]))
print(X1)
print(Y1)
plt.scatter(pointX,pointY)
plt.legend()
plt.show()
