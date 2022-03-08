import csv
import random
import math
import numpy as np

import algorithms as algs
 
def loadcsv(filename):
    #lines = csv.reader(open(filename, "rb"))
    with open(filename, "r", encoding="utf-8") as f:
      lines = csv.reader(f)
      dataset = list(lines)

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
 
def splitdataset(dataset, splitRatio):
    copy = list(dataset)
    trainsize = int(len(dataset) * splitratio)
    numinputs = len(dataset[0])-1
    Xtrain = np.zeros((trainsize,numinputs))
    ytrain = np.zeros(trainsize)
    for tt in range(trainsize):
        index = random.randrange(len(copy))
        vec = copy.pop(index)
        outputy = vec[-1]
        inputx = vec[0:numinputs]
        Xtrain[tt,:] = inputx
        ytrain[tt] = outputy

    testsize = len(copy)
    Xtest = np.zeros((testsize,numinputs))
    ytest = np.zeros(testsize)        
    for tt in range(testsize):
        vec = copy[tt]
        outputy = vec[-1]
        inputx = vec[0:numinputs]
        Xtest[tt,:] = inputx
        ytest[tt] = outputy
                       
    return ((Xtrain,ytrain), (Xtest,ytest))
 
def extract_features(dataset):
    """Implent this for Part 1: Question 2"""
    pass
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0
 
if __name__ == '__main__':
    filename = 'disease.csv'
    splitratio = 0.67
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset, splitratio)
    #print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), trainset[0].shape[0], testset[0].shape[0])
    print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows')
    classalgs = {'Random': algs.Classifier(),
                 'Naive Bayes': algs.NaiveBayes(),
                 'Logistic Regression': algs.LogitReg()
                 }
        
    for learnername, learner in classalgs.items():
        print('Running learner = ' + learnername)
        # Train model
        learner.learn(trainset[0], trainset[1])
        # test model
        predictions = learner.predict(testset[0])
        accuracy = getaccuracy(testset[1], predictions)
        print('Accuracy for ' + learnername + ': ' + str(accuracy))
 
