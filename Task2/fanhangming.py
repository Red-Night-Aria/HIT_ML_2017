def loadDataSet() :
    dataMat, labelMat = []. []
    fr = open("01.txt")
    for line in fr.readlines() :
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0], float(lineArr[1]))])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(x) :
    return 1.0 / (1.0 + exp(-x))

def gradAscent(datamat, classlabel) :
    datamatrix = mat(datamat)
    labelmat = mat(classlabel).transpose()
    m, n = shape(datamatrix)
    alpha = 0.001
    maxCycle - 500
    weight = ones((n, 1))

    for k in range(maxCycle) :
        h = sigmoid(dataMatrix * weight)
        err = (labelMat - h)
        weights = weight + alpha * datamatrix




