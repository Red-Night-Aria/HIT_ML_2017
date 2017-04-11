import numpy as np
import matplotlib.pyplot as plt
# 读取数据
X = [[float(a) for a in x.split()] for x in open('task3_x.dat').readlines()]
Y = [float(x) for x in open('task3_y.dat').readlines()]
# 生成矩阵
x_matrix = []
for x1, x2 in X:
    x_matrix.append((1, x1, x2, x1*x2, x1**2, x2**2, x1*x2**2, x2*x1**2,x1**3, x2**3, x1*x2**3, x2*x1**3, \
                     (x1**2)*(x2**2), x1**4, x2**4))
x_matrix = np.mat(x_matrix)
y_matrix = np.mat(Y).T
# 分类绘图
positive_x = []
negative_x = []
for x, y in zip(X, Y):
    if y > 0.5:
        positive_x.append(x)
    else:
        negative_x.append(x)
positive_x = np.array(positive_x)
negative_x = np.array(negative_x)
plt.scatter(positive_x[:, 0], positive_x[:, 1], c='b', marker='o', s=15)
plt.scatter(negative_x[:, 0], negative_x[:, 1], c='r', marker='x', s=15)
print(np.shape(x_matrix))
# 计算theta
theta = np.ones((15, 1))
alpha = 0.01
for i in range(50000):
    E = 1.0/(1.0 + np.exp(-x_matrix*theta)) - y_matrix
    theta = theta - alpha * x_matrix.T * E
# 绘制决策边界
a = np.arange(-4, 8, .01)
b = np.arange(-4, 4, .01)
a, b = np.meshgrid(a, b)
f = theta[0, 0] + theta[1, 0]*a + theta[2, 0]*b + theta[3, 0]*a*b + theta[4, 0]*a**2 + theta[5, 0]*b**2 + theta[6, 0]*a*b**2 + \
    theta[7, 0]*b*a**2 + theta[8, 0]*a**3 + theta[9, 0]*b**3 + theta[10, 0]*a*b**3 + theta[11, 0]*b*a**3 + theta[12, 0]*(a**2)*(b**2) + \
    theta[13, 0]*a**4 + theta[14, 0]*b**4
plt.contour(a, b, f, 0)
plt.show()
