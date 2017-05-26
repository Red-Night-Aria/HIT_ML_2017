# -*- coding:utf-8 -*-
'''
Created on 2017年4月27日

@author: 郑康杰
'''

import os
import struct
import numpy as np
from numba.tests.npyufunc.test_ufunc import dtype
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
from sklearn.neural_network import MLPClassifier

def LoadMNIST(kind = 'train'):
    with open('%s-labels.idx1-ubyte' % kind, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath,dtype = np.uint8)
    with open('%s-images.idx3-ubyte' % kind, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8,).reshape(len(labels), 784)
    return images, labels

X_train, Y_train = LoadMNIST()
X_test, Y_test = LoadMNIST(kind = 't10k')
fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
ax = ax.flatten()
for i in range(10):
    img = X_train[Y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
print("Running...")
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
clf.fit(X_train, Y_train)
print("Predicting...")
with open('result.txt','w') as OutputFile:
    pos = 0
    right = 0.0
    for i in X_test:
        pos += 1
        OutputFile.write("---------第%s组测试数据---------\n" % str(pos))
        r = clf.predict([i])[0]
        OutputFile.write("预测结果：%s\n" % str(r))
        OutputFile.write("答案：%s\n" % str(Y_test[pos-1]))
        if(Y_test[pos-1] == r):
            right += 1.0
    OutputFile.write("正确率：" + str((100.0*right)/float(pos)) + "%\n")
print("程序运行结束，结果已输出至result.txt文件中")
plt.show()