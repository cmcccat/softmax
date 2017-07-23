from numpy import *
from os import listdir
import random
import matplotlib.pyplot as plt
import pickle

#读取CIFAR-10数据集
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict
#将CIFAR-10解析为向量
def img2vector1():
    imgDataMat = []
    labelsMat = []
    for i in range(5):
        batch1 = unpickle("cifar-10-batches-py/data_batch_{}".format(i+1))
        for key in batch1:
            if (key.decode() == 'data'):
                imgDataMat.extend(batch1[key])
            if (key.decode() == 'labels'):
                labelsMat.extend(batch1[key])
    imgDataMat = array(imgDataMat)
    #将标签转换为向量
    m = len(labelsMat)
    matimg = mat(imgDataMat)
    matones = mat(ones((m,1)))
    imgrestdata = column_stack((matimg,matones))
    labelset = list(set(labelsMat))
    n = len(labelset)
    Labels = zeros((m,n))
    for i in range(m):
        if labelsMat[i] in labelset:
            Labels[i][labelset.index(labelsMat[i])] = 1
    return imgrestdata,Labels

def img2vector2():
    imgDataMat = []
    labelsMat = []
    batch1 = unpickle("cifar-10-batches-py/test_batch")
    for key in batch1:
        if (key.decode() == 'data'):
            imgDataMat.extend(batch1[key])
        if (key.decode() == 'labels'):
            labelsMat.extend(batch1[key])
    imgDataMat = array(imgDataMat)
    #将标签转换为向量
    m = len(labelsMat)
    matimg = mat(imgDataMat)
    matones = mat(ones((m, 1)))
    imgrestdata = column_stack((matimg, matones))
    labelset = list(set(labelsMat))
    n = len(labelset)
    Labels = zeros((m,n))
    for i in range(m):
        if labelsMat[i] in labelset:
            Labels[i][labelset.index(labelsMat[i])] = 1
    return imgrestdata,Labels

#定义sofmax函数
def softmax(inX):
    #0列1行
    maxVals = inX.max(1)
    inX -= maxVals
    einX = exp(inX)
    sumMat = sum(einX,axis=1)
    result = array(einX)/array(sumMat)
    return result


#数据归一化
def autoNorm(dataSet):
    #0:列最小，1:行最小
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    range = maxVals-minVals
    #zeros 属于array
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(range,(m,1))
    return normDataSet,range,minVals

#传统梯度下降法
def gradAscent(dataMatIn,classLabels,testDataMat,testlabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    m,n = shape(dataMatrix)
    mt,nt = shape(testDataMat)
    m1,n1 = shape(labelMat)
    theta = 0.000025
    maxCycles = 500  #最大迭代次数
    weights = ones((n,n1))
    weights1 = ones((n, n1))
    ax = plt.subplot(111)
    y1 = []
    y2 = []
    minbatch = int(500)
    selindex = int(m/minbatch)
    for k in range(maxCycles):
        weights1 = weights
        for iter in range(selindex):
            alpha = 1 / (1.0 + k + iter) + 0.0002
            randIndex = int(random.uniform(0, selindex))
            sta = max(randIndex*minbatch,0)
            end = min((randIndex+1)*minbatch,m)
            trainDataMat = dataMatIn[sta:end,:]
            trainlabelMat = labelMat[sta:end,:]
            h=softmax(trainDataMat * weights)
            error = (trainlabelMat - h)
            weights = weights + alpha * 1.0/minbatch * trainDataMat.transpose() * error-theta*weights
        errorCount1 = 0
        errorCount2 = 0
        for i in range(mt):
            # index = array(mat(weights)*mat(testDataMat[i]).T)
            index = array((mat(testDataMat[i]) * mat(weights)).T)
            result = (list(index).index(index.max(0)))
            if (result != list(testlabels[i]).index(1)): errorCount1 += 1
        for i in range(m):
            # index = array(mat(weights)*mat(testDataMat[i]).T)
            index = array((mat(dataMatIn[i]) * mat(weights)).T)
            result = (list(index).index(index.max(0)))
            if (result != list(classLabels[i]).index(1)): errorCount2 += 1
        # 绘制图像
        y1.append([1-float(errorCount2)/float(m)])
        y2.append([1-float(errorCount1)/float(mt)])
        ax.cla()
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_xlim(0, maxCycles + 10)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.plot(y1, label='train')
        ax.plot(y2, label='test')
        ax.legend(loc='best')
        plt.pause(0.05)
        print(k)
        print(sum(abs(weights - weights1)))
    return weights

#改进的随机梯度上升算法
def stocGrandAscent1(dataMatrix,classLabels,numIter = 1500):
    dataMatrix = mat(dataMatrix)
    labelMat = mat(classLabels)
    m, n = shape(dataMatrix)
    m1, n1 = shape(labelMat)
    weights = ones((n1,n))
    weights1 = ones((n1, n))
    ax = plt.subplot(111)
    y1 = []
    for j in range(numIter):
        weights1 = weights
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.001
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = softmax((weights*dataMatrix[randIndex].T).T)
            error = classLabels[randIndex] - h
            weights = weights + alpha * error.transpose() * dataMatrix[randIndex]+0.00000001*weights
            del (dataIndex[randIndex])
        #绘制图像
        y1.append([sum(abs(weights-weights1))])
        ax.cla()
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_xlim(0, numIter+10)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.plot(y1, label='train')
        ax.legend(loc='best')
        plt.pause(0.05)
        print(j)
        print(sum(abs(weights-weights1)))
    return weights

#将手写数字转换为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写数字识别测试代码
def handwritingfileLoad():
    hwLabels  = []
    #listdir 就是获取某一文件夹下文件名列表
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    #我们的测试数据组，将图片信息拓展成一个1024维的向量
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/{}".format(fileNameStr))
    # 将标签转换为向量
    m = len(hwLabels)
    matimg = mat(trainingMat)
    matones = mat(ones((m, 1)))
    imgrestdata = column_stack((matimg, matones))
    labelset = list(set(hwLabels))
    n = len(labelset)
    Labels = zeros((m, n))
    for i in range(m):
        if hwLabels[i] in labelset:
            Labels[i][labelset.index(hwLabels[i])] = 1
    return imgrestdata, Labels

def handwritingfileLoad1():
    hwLabels  = []
    #listdir 就是获取某一文件夹下文件名列表
    trainingFileList = listdir('testDigits')
    m = len(trainingFileList)
    #我们的测试数据组，将图片信息拓展成一个1024维的向量
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("testDigits/{}".format(fileNameStr))
    # 将标签转换为向量
    matimg = mat(trainingMat)
    matones = mat(ones((m, 1)))
    imgrestdata = column_stack((matimg, matones))
    m = len(hwLabels)
    labelset = list(set(hwLabels))
    n = len(labelset)
    Labels = zeros((m, n))
    for i in range(m):
        if hwLabels[i] in labelset:
            Labels[i][labelset.index(hwLabels[i])] = 1
    return imgrestdata, Labels

def testhwclass():
    imgDataMat, labelsMat = img2vector1()
    testDataMat, testlabelsMat = img2vector2()
    weights = gradAscent(imgDataMat, labelsMat,testDataMat,testlabelsMat)
    errorCount = 0.0
    mTest = len(testlabelsMat)
    for i in range(mTest):
        #index = array(mat(weights)*mat(testDataMat[i]).T)
        index = array((mat(testDataMat[i])*mat(weights)).T)
        result = (list(index).index(index.max(0)))
        if(result != list(testlabelsMat[i]).index(1)):errorCount += 1
    print("the total number of errors is；{}".format(errorCount))
    print("the tota; error rate is:{}".format(errorCount / float(mTest)))



