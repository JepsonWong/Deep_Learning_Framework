#encoding:utf-8

import os
import math
import numpy

# 从文件夹中读取文件生成特征矩阵和类别矩阵
def loadData(dir):
    trainfileList = os.listdir(dir)
    m = len(trainfileList)
    dataArray = numpy.zeros((m, 1024))  # 构造m*1024的特征矩阵
    labelArray = numpy.zeros((m, 1)) # 构造类别矩阵

    for i in range(m):
        returnArray = numpy.zeros((1, 1024))  # 生成单个文件的特征
        filename = trainfileList[i]
        fr = open('%s/%s' % (dir, filename))
        for j in range(32):
            lineStr = fr.readline()
            for k in range(32):
                returnArray[0, 32 * j + k] = int(lineStr[k])
        dataArray[i, :] = returnArray  # 存储单个文件的特征
        filename0 = filename.split('.')[0]
        label = filename0.split('_')[0]
        labelArray[i] = int(label)  # 存储类别
    return dataArray, labelArray

# 实现sigmoid函数
def sigmoid(x):
    print x
    m, n = numpy.shape(x)
    x_new = []
    for i in range(m):
        # print x[i,0]
        x_new.append(1.0 / (1 + math.exp(-x[i, 0])))
    y = numpy.mat(x_new).transpose()
    #print numpy.shape(y)
    return y
    #return 1.0 / (1 + math.exp(-x))

# 梯度下降算法，alpha为步长，maxCycles为迭代次数。
def gradDescent(train, label, alpha, maxCycles):
    dataMat = numpy.mat(train) # m*n的矩x阵
    labelMat = numpy.mat(label) # m*1的矩阵
    m, n = numpy.shape(dataMat)
    weigh = numpy.random.normal(0.0, 0.1, (n,1)) #weigh = numpy.ones((n, 1)) # n*1的参数
    print weigh
    for i in range(maxCycles):
        h = sigmoid(dataMat * weigh)
        error = labelMat - h  # m*1的矩阵，本来应该是预测的标记减去真实的标记，这里真实的标记减去预测的标记，所以下面alpha之前为加号。要不然就是减号，因为是梯度下降。
        weigh = weigh + alpha * dataMat.transpose() * error # n*1 = n*1 + 常数*(n*m)*(m*1)
    return weigh

# 根据训练的模型对输入的样本进行预测
def classfy(testdir, weigh):
    dataArray, labelArray = loadData(testdir)
    dataMat = numpy.mat(dataArray)
    labelMat = numpy.mat(labelArray)
    h = sigmoid(dataMat*weigh)  # (m*n)*(n*1)=m*1
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print int(labelMat[i]), 'is classfied as: 1'
            if int(labelMat[i]) != 1:
                error += 1
                print 'error'
        else:
            print int(labelMat[i]), 'is classfied as: 0'
            if int(labelMat[i]) != 0:
                error += 1
                print 'error'
    print 'error rate is:','%.4f' %(error/m)

def digitRecognition(trainDir, testDir, alpha = 0.01, maxCycles = 10):
    train, label = loadData(trainDir)
    weigh = gradDescent(train, label, alpha, maxCycles)
    print weigh
    classfy(testDir, weigh)

if __name__ == '__main__':
    #x = numpy.mat([[1], [2], [3], [4], [5]])
    #sigmoid(x)
    digitRecognition("./train", "./test")