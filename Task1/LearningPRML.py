'''
Created on 2017年3月23日

@author: 郑康杰
'''
from comtypes.test.test_safearray import com_refcnt
import numpy as np
import matplotlib.pyplot as plt
from turtledemo.__main__ import font_sizes

file1=open('q2x.txt')
dataX=file1.readlines()
file2=open('q2y.txt')
dataY=file2.readlines()

file1.close()
file2.close()

X=[]
Y=[]

for eachline in dataX:
    X.append(float(eachline))

for eachline in dataY:
    Y.append(float(eachline))

plt.figure(figsize=(8,4))
plt.scatter(X,Y,s=50,color='r')

epsilon = 0.00000000001

# 学习率
alpha = 0.03
max_itor = 2000
cnt = 0
ok = 0
m = len(X)
Points_Number = 20

theta0 = 0
theta1 = 0

def h(x):
    return (theta0+theta1*x)

def draw(color='b',width = 0):
    x=np.linspace(-6, 13, Points_Number)
    y=[]
    for i in range(Points_Number):
        y.append(h(x[i]))
    if  width == 0:
        plt.plot(x,y,color,linewidth = 2)
    else:
        plt.plot(x,y,color,linewidth = 4)
    

pj = 0.0
while True :
    cnt += 1
    der1 = 0.0
    der2 = 0.0
    for i in range(m):
        der1 += (h(X[i]) - Y[i]) 
        der2 += (h(X[i]) - Y[i])*X[i]
    der1 /= m;
    der2 /= m;
    theta0 -= alpha*der1
    theta1 -= alpha*der2
    if cnt % 7 == 0 :
        draw()
    j = 0.0
    for i in range(m):
        j += (h(X[i]) - Y[i])**2
    j /= (2*m)
    if abs(pj -j) < epsilon:
        ok += 1
    else:
        ok = 0
    if ok > 10:
        break;
    if cnt >  max_itor :
        break
    pj = j
    
plt.title("TIMES:" + str (cnt) + "     The green line represents the final result.")
plt.legend()
draw('g',1)
plt.show()

    
    