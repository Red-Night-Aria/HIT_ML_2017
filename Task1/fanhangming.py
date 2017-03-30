from numpy import *
from pylab import *

X = [(float)(x.strip()) for x in open('q2x.dat').readlines()]
Y = [(float)(y.strip()) for y in open('q2y.dat').readlines()]
scatter(X, Y, linewidth = 0.1)
xx = linspace(min(X), max(X), 256, endpoint = True)
a, b, u, w, pre = 0, 0, 0.01, 0.01, -1

for j in range(0, 10000):
    da, db, now = 0.00, 0.00, 0.00
    for i in range(0, 100) :
        da += X[i] * (b + a * X[i] - Y[i])
        db += (b + a * X[i] - Y[i])
    a -= u * (da / 100)
    b -= u * (db / 100)
    for i in range(0, 100) :
        now += fabs(b + a * X[i] - Y[i])
    yy = b + a * xx
    plot(xx, yy)
    if fabs(now - pre) < w :
        break
    pre = now
title('fanhangming')
show()
savefig("fanhangming.png")