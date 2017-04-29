import numpy as np
import matplotlib.pyplot as plt
from numpy import *

file1 = open("task3_x.dat")
dataX = file1.readlines()
file2 = open("task3_y.dat")
dataY = file2.readlines()

file1.close()
file2.close()

X = []
Y = []

for eachline in dataX:
    X1 = float(eachline.split()[0])
    X2 = float(eachline.split()[1])
    X.append((1,X1,X2,X1*X2,X1**2,X2**2,X1*X2**2,X2*X1**2,X1**3,X2**3,X1*X2**3,X2*X1**3,X1**2*X2**2,X1**4,X2**4))
for eachline in dataY:
    Y.append(float(eachline))

X = mat(X)
Y = mat(Y).T

theta = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
theta = mat(theta).T
m = size(X1)
alpha = 0.01
Max_step = 80000

for i in range(Max_step):
    error = 1/(1 + exp(-X*theta)) - Y
    theta = theta - alpha * X.T * error
print(theta)
x1 = np.arange(-5, 9, .01)
x2 = np.arange(-5, 5, .01)
x1,x2 = np.meshgrid(x1,x2)
f = theta[0,0] + theta[1,0]*x1 + theta[2,0]*x2 + theta[3,0]*x1*x2 + theta[4,0]*x1**2 + theta[5,0]*x2**2 + theta[6,0]*x1*x2**2 + theta[7,0]*x2*x1**2 + theta[8,0]*x1**3 + theta[9,0]*x2**3 + theta[10,0]*x1*x2**3 + theta[11,0]*x2*x1**3 + theta[12,0]*(x1**2)*(x2**2) + theta[13,0]*x1**4 + theta[14,0]*x2**4

plt.contour(x1,x2,f,0)

t = 0;
for each in Y:  
    if each > 0:
        plt.scatter(X[t,1],X[t,2],color = 'red',marker = "x")
    else:
        plt.scatter(X[t,1],X[t,2],color = 'blue')
    t = t + 1
plt.title("cswyxiang's Graphics")
plt.legend()
plt.show()