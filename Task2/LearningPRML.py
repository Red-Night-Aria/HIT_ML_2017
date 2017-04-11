'''
Created on 2017Äê3ÔÂ28ÈÕ

@author: Ö£¿µ½Ü
'''
from numpy import *
import matplotlib.pyplot as plt

X = []
Y = []
K = [0.01,0.03,0.1,0.3,1.0,3.0,10.0,100.0]
COLOR_LIST = ['gold','brown','peru','greenyellow','gray','blueviolet','tomato','olive']

def LoadDataSet():
    file1=open('q2x.dat')
    dataX=file1.readlines()
    file2=open('q2y.dat')
    dataY=file2.readlines()
    file1.close()
    file2.close()
    for eachline in dataX:
        X.append(float(eachline))
    
    for eachline in dataY:
        Y.append(float(eachline))

def predict(x,k):
    xMat = mat(X).T
    yMat = mat(Y).T
    m = xMat.shape[0]
    W = eye(m)
    for i in range(m):
        W[i,i] = exp((-(xMat[i,0]-x)**2)/(2*k**2))
    V = xMat.T * W * xMat
    if linalg.det(V) == 0.0:
        print("Wrong!")
        return 0
    theta = V.I * xMat.T * W * yMat
    ans = theta * x
    return ans[0,0]

LoadDataSet()
X1 = []
for x in X:
    X1.append(x)
X1.sort(key=None, reverse=False)
Y1 = []
for i in range(8):
    Y_ans = []
    for x in X1:
        Y_ans.append(predict(x,K[i]))
    plt.plot(X1,Y_ans,color = COLOR_LIST[i],label = 'K = ' + str(K[i]))
for x in X1:
    pos = -1
    for xx in X:
        pos += 1
        if xx==x:
            Y1.append(Y[pos])
            break
plt.scatter(X1,Y1)
plt.legend()
plt.show()

    