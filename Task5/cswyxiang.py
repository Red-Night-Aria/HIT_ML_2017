import numpy as np  
import struct  
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def loadImageSet(filename):  
    binfile= open(filename, 'rb')  
    buffers = binfile.read()   
    head = struct.unpack_from('>IIII' , buffers ,0)  
    offset = struct.calcsize('>IIII')  
    imgNum = head[1] 
    width = head[2]  
    height = head[3]

    bits = imgNum * width * height  
    bitsString = '>' + str(bits) + 'B'
   
    imgs = struct.unpack_from(bitsString,buffers,offset)
   
    binfile.close()  
    imgs = np.reshape(imgs,[imgNum,1,width*height]) 

    return imgs
   
def loadLabelSet(filename):  
    binfile = open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>II' , buffers ,0)  
    imgNum=head[1]  
   
    offset = struct.calcsize('>II')  
    numString = '>'+str(imgNum)+"B"  
    labels = struct.unpack_from(numString , buffers , offset)  
    binfile.close()  
    labels = np.reshape(labels,[imgNum,1])  
   
    return labels
   

train_imgs = loadImageSet("train-images.idx3-ubyte")  
train_labels = loadLabelSet("train-labels.idx1-ubyte") 
test_imgs = loadImageSet("t10k-images.idx3-ubyte") 
test_labels = loadLabelSet("t10k-labels.idx1-ubyte") 

#观察待训练数据
#for i in range(10):
#	print(np.shape(imgs[i]))
#	img = np.reshape(test_imgs[i],[28,28])
#	fig = plt.figure()
#	plotwindow = fig.add_subplot(111)
#	plt.imshow(img,cmap='gray')
#	plt.show()

print(np.shape(train_imgs))
print(np.shape(train_labels))

clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1)
clf.fit(train_imgs[:,0,:],train_labels)

test_num = np.shape(test_imgs)[0]

rights = 0;
f = open('test_result.txt', 'w')
for i in range(test_num):
	if(clf.predict(test_imgs[i])==test_labels[i]):
		rights+=1
	f.write("test%s    predict: %s    result: %s\n" % (str(i),str(int(clf.predict(test_imgs[i]))),str(int(test_labels[i]))))

f.write("Correct rate:  %s %%" % str(float(rights)*100/test_num))
f.close()


