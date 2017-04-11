# -*- coding:utf-8 -*-
#最小二乘法一次函数形式实现

import numpy as np
import matplotlib.pyplot as plt

#读入X,Y数据
X = [(float)(x.strip()) for x in open('q2x.dat').readlines()]
Y = [(float)(y.strip()) for y in open('q2y.dat').readlines()]

#直线拟合形式为y = a + bx
#设N为点的个数
#a = ((sum(xi^2))*(Sum(yi))-(sum(xi))*(sum(xi*yi)))/(N*(sum(xi^2))-(sum(xi))^2)
#b = (N*(sun(xi*yi))-(sum(xi))(sum(yi)))/(N*(sum(xi^2))-(sum(xi)^2))
#通过观察公式可得里面的数据有重合
#设 x2_sum=sum(xi^2) xy_sum=sum(xi*yi) x_sum=sum(xi) y_sum=sum(yi)

x2_sum = sum([px*px for px in X])
xy_sum = sum([px*py for px,py in zip(X,Y)])
x_sum = sum(X)
y_sum = sum(Y)
N = len(X)

a = (x2_sum * y_sum - x_sum * xy_sum)/(N * x2_sum - x_sum * x_sum)
b = (N * xy_sum - x_sum * y_sum)/(N * x2_sum - x_sum * x_sum)

#得到函数y = a + bx
def func(L):
    return [(a + b * x) for x in L]

#设置X,Y轴的上下限
plt.xlim(-5.2,13.0)
plt.ylim(-5.0,5.0)

#生成绘制点
n=2
PointsX = np.linspace(-5.5,13.0,n,endpoint=True)
PointsY = func(PointsX)

#散点绘制
plt.scatter(X,Y,c='r', s=10, marker='o')

#所得函数绘制
plt.plot(PointsX,PointsY)

#设置标题
plt.title("AlongWY's Graphics")

#显示界面
plt.show()

#保存图片
plt.savefig("AlongWY.png")