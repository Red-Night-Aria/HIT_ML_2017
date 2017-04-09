import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def sig(x):
    return 1.0 / (1 + np.exp(-x))

tX = np.array([[float(y.strip()) for y in x.split()] for x in open("task3_x.dat").readlines()])
Y = np.array([float(x.strip()) for x in open("task3_y.dat").readlines()])

order = 3
X = np.matrix([[x[0]**a * x[1]**b for a in range(order+1) for b in range(order-a+1)] for x in tX])
theta = np.zeros((order+1)*(order+2)//2)

rate = 1e-1
for i in range(20000):
    A = theta * X.T
    E = sig(A) - Y    
    theta = theta - rate * E * X    

x = np.linspace(min(tX[:,0]), max(tX[:,0]), 1e2)
y = np.linspace(min(tX[:,1]), max(tX[:,1]), 1e2)
x, y = pl.meshgrid(x, y)
f = np.zeros(x.shape)
theta = np.array(theta)
t = 0
for a in range(order+1):
    for b in range(order+1-a):
        f += theta[0][t] * (x**a) * (y**b)
        t += 1

plt.figure(figsize=(20,15))    
plt.contour(x, y, f, 0);
for i, p in enumerate(tX):
    plt.scatter(p[0], p[1], c='r' if Y[i] == 1.0 else 'b', s=200, marker='o')
plt.title("MMMMMua's Graphics", fontsize=40)
plt.show()
plt.savefig("MMMMMua.png", dpi=300)
