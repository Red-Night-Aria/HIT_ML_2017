# -*- coding:utf-8 -*-
#   神经网络拟合

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import random

#   加载数据
file = open('watermelon.txt')
lines = file.readlines()
lines.pop(0)
data = []
label = []
for line in lines:
    curr = line.split()
    data.append([float(curr[1]), float(curr[2])])
    label.append(float(curr[3] == '是'))


# 定义 sigmoid 函数
def sigmoid(in_x):
    return 1.0 / (1.0 + np.exp(-in_x))


# 定义随机生成值函数
# 生成 m * n 的矩阵
def random_mn(m, n):
    res = []
    for i in range(m):
        res.append([])
        for j in range(n):
            res[i].append(random.random())
    return res


def random_n(n):
    res = []
    for i in range(n):
        res.append(random.random())
    return res


# 散列点绘制
for i in range(len(label)):
    if label[i] == 1.0:
        plt.scatter(data[i][0], data[i][1], c='r', marker='+')
    else:
        plt.scatter(data[i][0], data[i][1], c='b')

# 数据向量化
data = np.mat(data)
label = np.mat(label).T

# 定义输入层,输出层和隐层的个数
output_num = 1
hidden_num = 3
input_num = 2

# 输出层和隐层阈值
output_threshold = np.mat(random_n(output_num))
hidden_threshold = np.mat(random_n(hidden_num))

# 输出层的权值
output_weight = np.mat(random_mn(hidden_num, output_num))
# 隐层的权值
hidden_weight = np.mat(random_mn(input_num, hidden_num))

# 迭代次数
cycle = 50000

# 学习速率
alpha = 0.1

for time in range(cycle):
    for i in range(len(data)):
        hidden_in = data[i] * hidden_weight
        hidden_out = sigmoid(hidden_in - hidden_threshold)
        output_in = hidden_out * output_weight
        output_out = sigmoid(output_in - output_threshold)

        hidden_out_array = np.asarray(hidden_out)

        output_threshold_update = output_out * (1 - output_out) * (np.asarray(label[i]) - output_out)
        output_weight_update = output_threshold_update * hidden_out_array

        hidden_threshold_update = hidden_out_array * (1 - hidden_out_array) * np.asarray((output_weight * output_threshold_update).T)
        hidden_weight_update = data[i].T * hidden_threshold_update

        output_threshold -= alpha * output_threshold_update
        output_weight += alpha * output_weight_update.T

        hidden_threshold -= alpha * hidden_threshold_update
        hidden_weight += alpha * hidden_weight_update

# 显示结果
print(output_weight)
print(output_threshold)
print(hidden_weight)
print(hidden_threshold)

x1 = np.arange(0, 0.85, .01)
x2 = np.arange(0, 0.55, .01)
x1, x2 = np.meshgrid(x1, x2)


f = sigmoid(
    sigmoid(x1 * hidden_weight[0, 0] + x2 * hidden_weight[1, 0] - hidden_threshold[0,0]) * output_weight[0, 0] +
    sigmoid(x1 * hidden_weight[0, 1] + x2 * hidden_weight[1, 1] - hidden_threshold[0,1]) * output_weight[1, 0] +
    sigmoid(x1 * hidden_weight[0, 2] + x2 * hidden_weight[1, 2] - hidden_threshold[0,2]) * output_weight[2, 0] -
    output_threshold
)

plt.contour(x1, x2, f, 0)

# 显示界面
plt.show()
