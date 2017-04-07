import numpy as np
import matplotlib.pyplot as plt
from numpy import *

file1 = open("task3_x.dat")
dataX = file1.readlines()
file2 = open("task3_y.dat")
dataY = file2.readlines()

file1.close()
file2.close()

X1 = []
X2 = []
Y = []

for eachline in dataX:
	X1.append(float(eachline.split()[0]))
	X2.append(float(eachline.split()[1]))
for eachline in dataY:
	Y.append(float(eachline))

def h(x1,x2,theta):
	return theta[0] + theta[1]*x1 + theta[2]*x2 + theta[3]*x1**2 + theta[4]*x2**2 + theta[5]*x1**3 + theta[6]*x2**3 + theta[7]*x1*x2 + theta[8]*x1*x2**2 + theta[9]*x2*x1**2 + theta[10]*x1*x2**3+theta[11]*x2*x1**3 + theta[12]*x2**2*x1**2 +theta[13]*x1**4+ theta[14]*x2**4

theta = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
m = size(X1)
alpha = 0.01
Max_step = 60000

for i in range(Max_step):
	sigma = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for j in range(m):
		sigma[0] = sigma[0] + 1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j] 
		sigma[1] = sigma[1] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]
		sigma[2] = sigma[2] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]
		sigma[3] = sigma[3] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]**2
		sigma[4] = sigma[4] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]**2
		sigma[5] = sigma[5] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]**3
		sigma[6] = sigma[6] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]**3
		sigma[7] = sigma[7] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]*X1[j]
		sigma[8] = sigma[8] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]*X2[j]**2
		sigma[9] = sigma[9] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]*X1[j]**2
		sigma[10] = sigma[10] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]*X2[j]**3
		sigma[11] = sigma[11] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]*X1[j]**3
		sigma[12] = sigma[12] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]**2*X2[j]**2
		sigma[13] = sigma[13] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X1[j]**4
		sigma[14] = sigma[14] + (1/(1+exp(-h(X1[j],X2[j],theta)))-Y[j])*X2[j]**4
	for k in range(15):
		theta[k] = theta[k] - alpha*(1/m)*sigma[k]

x1 = np.arange(-5, 9, .01)
x2 = np.arange(-5, 5, .01)
x1,x2 = np.meshgrid(x1,x2)
y = h(x1,x2,theta)

plt.contour(x1,x2,y,0)

t = 0;
for each in Y:	
	if float(each) > 0:
		plt.scatter(X1[t],X2[t],color = 'red',marker = "x")
	else:
		plt.scatter(X1[t],X2[t],color = 'blue')
	t = t + 1
plt.title("cswyxiang's Graphics")
plt.legend()
plt.show()
