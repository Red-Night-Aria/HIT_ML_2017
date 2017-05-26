import numpy as np
import struct
from sklearn.neural_network import MLPClassifier
from PIL import Image
import os, glob


def read_image(filename):
    im = 255-np.array((Image.open(filename).resize((28, 28)).convert('L')), 'f')
    im = im.reshape((1, 28*28))
    return im


def judge_image(file):
    filepath, filename = os.path.split(file)
    try_x = read_image(file)
    try_y = int(filename[0])
    num = 0
    result = clf.predict(try_x)
    for i in result[0]:
        if i == 1:
            return num, try_y
        num += 1
    return 'skip', try_y


def load_idx3(idx3_file):
    bin_data = open(idx3_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((1, num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def load_idx1(idx1_file):
    bin_data = open(idx1_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty((num_images, 10))
    for i in range(num_images):
        labels[i][struct.unpack_from(fmt_image, bin_data, offset)[0]] = 1
        offset += struct.calcsize(fmt_image)
    return labels


Train_x = load_idx3('train-images.idx3-ubyte')
Train_y = load_idx1('train-labels.idx1-ubyte')
Test_x = load_idx3('t10k-images.idx3-ubyte')
Test_y = load_idx1('t10k-labels.idx1-ubyte')
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(60, 60, 60), random_state=1)
clf.fit(Train_x, Train_y)
err_num = 0
(num_test, size) = np.shape(Test_x)
for i in range(num_test):
    if not (clf.predict([Train_x[i]]) == list(Train_y[i])).all():
        err_num += 1

with open('output.txt', 'w') as output:
    output.write("Test set's error rate: %s%%\n" % str((float(err_num*100)/float(num_test))))
    err_num = 0
    skip_num = 0
    size = len(glob.glob('D:\\Machine Learning\\HIT_ML_2017\\Task5\\my_handwriting\\*.JPG'))
    for file in glob.glob('D:\\Machine Learning\\HIT_ML_2017\\Task5\\my_handwriting\\*.JPG'):
        output.write('excepted:  %s         got:  %s\n' % (str(judge_image(file)[1]), str(judge_image(file)[0])))
        if judge_image(file)[0] == 'skip':
            skip_num += 1
        elif not judge_image(file)[0] == judge_image(file)[1]:
            err_num += 1
    output.write("My set's error rate: %s%%\n" % str((float(err_num*100)/float(size))))
    output.write("skip_num: %s        err_num: %s" % (str(skip_num), str(err_num)))
