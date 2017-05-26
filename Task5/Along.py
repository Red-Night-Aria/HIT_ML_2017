# -*- coding:utf-8 -*-
from PIL import Image
import struct
# 导入矩阵运算库
import numpy as np
# 导入机器学习库
from sklearn.neural_network import MLPClassifier
# 保存训练模型
from sklearn.externals import joblib

img_path = "train-images.idx3-ubyte"
label_path = "train-labels.idx1-ubyte"

test_img = "t10k-images.idx3-ubyte"
test_label = "t10k-labels.idx1-ubyte"


class Mnist:
    def __init__(self, m_image_path, m_label_path):
        # 读取图片和标签
        file_label = open(m_label_path, 'rb')
        file_image = open(m_image_path, 'rb')

        # 读取Magic Number
        self.label_magic_number = struct.unpack('>i', file_label.read(4))[0]
        self.image_magic_number = struct.unpack('>i', file_image.read(4))[0]

        # 读取数据数目
        self.label_num = struct.unpack('>i', file_label.read(4))[0]
        self.image_num = struct.unpack('>i', file_image.read(4))[0]
        if self.label_num != self.image_num:
            print("Data Error!!!")

        # 读取图片像素行列数
        self.pix_rows = struct.unpack('>i', file_image.read(4))[0]
        self.pix_lines = struct.unpack('>i', file_image.read(4))[0]

        # 读取标签
        self.label = []
        for i in range(self.label_num):
            self.label.append(struct.unpack('>i', b'\x00\x00\x00' + file_label.read(1))[0])
        file_label.close()

        self.image = []
        for i in range(self.image_num):
            im = bytearray(file_image.read(self.pix_rows * self.pix_lines))
            self.image.append(np.array(im))
        file_image.close()

    def display_image(self):
        return [x.reshape(self.pix_rows, self.pix_lines) for x in self.image]


# mnist = Mnist(img_path, label_path)
test = Mnist(test_img, test_label)

# clf = svm.SVC(gamma=0.01, C=100, max_iter=5000)
# clf.fit(mnist.image, mnist.label)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 60, 60), random_state=1)
# clf.fit(mnist.image, mnist.label)

# 模型保持
# joblib.dump(clf, "train_model.m")
# 模型重调
clf = joblib.load("train_model.m")

Predict = clf.predict(test.image)

correct = 0

output = open("AlongTest.txt", 'w')
for i in range(test.image_num):
    print("正确结果:{0} 预测结果:{1}".format(test.label[i], Predict[i]))
    output.write("正确结果:{0} 预测结果:{1}\n".format(test.label[i], Predict[i]))
    if test.label[i] == Predict[i]:
        correct += 1

output.write("正确率:{}\n".format(correct / test.image_num))
