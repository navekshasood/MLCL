import csv
import random
import math
import numpy as np
import algorithms as algs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
 
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
 
def loadIMDB(filename):
  dataset = pd.read_csv(filename)
  return dataset

def convert_labels(label):
  if label == 'positive':
    return 1
  else: # label = 'negative'
    return 0

def extract_features(X_train, X_test, X_0, X_1):
    """Implent this for Part 1: Question 2"""
    count = CountVectorizer(encoding = 'utf-8', strip_accents='unicode', ngram_range=(1,2), stop_words='english', max_features=5000)
    bow_train = count.fit_transform(X_train)
    bow_test = count.transform(X_test)
    bow_train_0 = count.transform(X_0)
    bow_train_1 = count.transform(X_1)
    return bow_train, bow_test, bow_train_0, bow_train_1

def separate_classes(X, y):
  ## for naive bayes
  class_0 = []
  class_1 = []
  for row in range(X.shape[0]):
    if y[row] == 0:
      class_0.append(X[row])
    elif y[row] == 1:
      class_1.append(X[row])
  return class_0, class_1


def splitIMDB(df):

  df['sentiment'] = df['sentiment'].apply(convert_labels)
  
  X = df.review
  y = df.sentiment

  X_mat = X.to_numpy()
  y_mat = y.to_numpy()

  X_train, X_test, y_train, y_test = train_test_split(X_mat, y_mat, test_size=.2)

  #print(X_train[0])
  X_train_0, X_train_1 = separate_classes(X_train, y_train)
  X_train_f, X_test_f, X_train_0_f, X_train_1_f = extract_features(X_train, X_test, X_train_0, X_train_1)

  #print('BOW_train:', X_train_f.shape)
  #print('BOW_test:', X_test_f.shape)

  return (X_train_f,y_train), (X_test_f, y_test), X_train_0_f, X_train_1_f

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0
 
if __name__ == '__main__':

    ## uncomment the desired data file
    filename = 'disease.csv'
    #filename = 'IMDB_Dataset.csv'

    if filename == 'disease.csv':
      splitratio = 0.67
      dataset = loadcsv(filename)
      trainset, testset = splitdataset(dataset, splitratio)
      print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows')
      classalgs = {'Random': algs.Classifier(),
                  'Naive Bayes': algs.NaiveBayes('disease'),
                  'Logistic Regression': algs.LogitReg()
                  }

    elif filename == 'IMDB':
      dataset = loadIMDB(filename)
      trainset, testset, class_0, class_1 = splitIMDB(dataset)
      print(class_0.get_shape())
      print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows')
      classalgs = {'Random': algs.Classifier(),
                  'Naive Bayes': algs.NaiveBayes('IMDB', class_0, class_1),
                  'Logistic Regression': algs.LogitReg()
                  }

    #print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), trainset[0].shape[0], testset[0].shape[0])
        
    for learnername, learner in classalgs.items():
        print('Running learner = ' + learnername)
        # Train model
        learner.learn(trainset[0], trainset[1])
        # test model
        predictions = learner.predict(testset[0])
        accuracy = getaccuracy(testset[1], predictions)
        print('Accuracy for ' + learnername + ': ' + str(accuracy))
 
