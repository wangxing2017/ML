from numpy import *
import operator

def createDataSet():
    pass
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape(0)
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        vote = labels[sortedDistIndicies[i]]
        classCount[vote] = classCount.get(vote,0)+1
        sortedclassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedclassCount[0][0]
    
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect [0,32*i+j] = int(lineStr[j])
        return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range (m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(fileNameStr)
    testFileList = listdir('')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        
        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,classNumStr))
        
        if (classifierResult != classNumStr): errorCount += 1
    print("the total number of errors is: %d"%errorCount)
    print("the total error rate is:%f"%(errorCount/float(mTest)))     