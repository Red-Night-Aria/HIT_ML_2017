import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pylab as pl

traits    = []
judge     = []
# 3 neurons
In_param  = np.array(npr.random(size = (2, 3)))
Out_param = np.array(npr.random(size = (3, 1)))
thres_out = npr.random(size = (1, 1))
thres_in  = npr.random(size = (1, 3))


def Data_Processing():
	data = open("watermelon.txt").readlines()[1:]
	for lines in data:
		elements = lines.split()
		traits.append([float(x) for x in elements[1:-1]])
		judge.append(1.0 if elements[-1] == "是" else 0.0)

def sigmoid(In):
	for x in In: x = 1.0 / (1 + np.exp(-x))
	return x

def Back_Propagation():
	global In_param
	global Out_param
	global thres_in
	global thres_out

	rate = 0.1

	for epoch in range(50000):
		for id, item in enumerate(traits):

			hid_In    = np.array(np.mat(item) * np.mat(In_param))
			hid_Out   = sigmoid(hid_In - thres_in)

			fin_In    = np.array(np.mat(hid_Out) * np.mat(Out_param))
			fin_Out   = sigmoid(fin_In - thres_out)

			g         = fin_Out * (1.0 - fin_Out) * (judge[id] - fin_Out)
			e         = hid_Out * (1.0 - hid_Out) * np.array([np.dot(x, g) for x in Out_param])

			In_param  += np.array(rate * np.matrix(item).T * np.matrix(e))
			Out_param += np.array(rate * np.matrix(hid_Out).T * np.matrix(g))
			thres_in  -= rate * e
			thres_out -= rate * g

def f(x, y):
	ret = []
	X   = x.reshape(1, 10000)
	Y   = y.reshape(1, 10000)

	for id in range(10000):
		In    = np.matrix([X[0,id],Y[0,id]])
		H_in  = In * np.matrix(In_param)
		H_out = sigmoid(np.array(H_in - thres_in))
		F_in  = H_out * np.matrix(Out_param)
		F_out = sigmoid(np.array(F_in - thres_out))
		ret.append(F_out[0])

	ret = np.array(ret)
	return ret.reshape(100, 100)

def Output():
	x1     = np.linspace(0, 0.85, 1e2)
	x2     = np.linspace(0, 0.55, 1e2)
	x1, x2 = pl.meshgrid(x1, x2)

	plt.figure(figsize = (20, 15))

	for id, item in enumerate(traits):
		plt.scatter(item[0], item[1], c = 'r' if judge[id] == 1.0 else 'b', s = 200, marker = 'o')

	plt.contour(x1, x2, f(x1, x2), 0)
	plt.show()
	plt.savefig("MMMMMMMua.png", dpi = 300)

Data_Processing()
Back_Propagation()
Output()
