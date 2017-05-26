# -*- coding: utf-8 -*-
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from PIL import Image

def load_mnist(kind='train'):
	with open('%s-labels.idx1-ubyte' % kind, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype = np.uint8)
	with open('%s-images.idx3-ubyte' % kind, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
		images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
	return images, labels

X_train, Y_train = load_mnist()
X_test, Y_test = load_mnist(kind = 't10k')
print('Multi-layer Perceptron Building...')
#clf = MLPClassifier(alpha = 0.00001, hidden_layer_sizes = (100,100))
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=1)
clf.fit(X_train, Y_train)
print('Done!')


with open('Chanpagne.txt', 'w') as Chanpagne:
	pos = 0
	right = 0.0
	for i in X_test:
		pos += 1
		temp = clf.predict([i])[0]
		if(Y_test[pos-1] == temp):
			right += 1.0
	Chanpagne.write("MNIST训练集的正确率：" + str((100.0*right)/float(pos)) + "%\n") 

	Chanpagne.write("偷用狗爷的手写体数据测试XD：\n")
	right2 = 0
	for i in range(100):
		img = Image.open("C:\\handwriting\\%s.jpg" % str(i)).convert('L')
		new_img = 255 - np.array(img).reshape(784, )
		ans = clf.predict([new_img])[0]
		if(ans == (int)(i/10)):
			right2 += 1
		Chanpagne.write("Real:             %s             predict:          %s\n" % (str((int)(i/10)), str(ans)))
	Chanpagne.write('狗爷手写体的准确率：      %s%%\n' % str(right2))
	Chanpagne.write("我的手写体数据测试：\n")
	right3 = 0
	for i in range(10):
		img = Image.open("C:\\my_handwriting\\%s.png" % str(i)).convert('L')
		new_img = 255 - np.array(img).reshape(784, )
		ans = clf.predict([new_img])[0]
		if(ans == i):
			right3 += 1
		Chanpagne.write("Real:             %s             predict:          %s\n" % (str(i), str(ans)))
	Chanpagne.write('我的手写体的准确率：      %s%%' % str(right3*10))
