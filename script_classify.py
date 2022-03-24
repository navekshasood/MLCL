import csv
import random
import math
import numpy as np
import algorithms as algs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
 
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
    trainsize = int(len(dataset) * splitRatio)
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
 
##################
#      IMDB      #
##################

def loadIMDB(filename):
  ## load IMDB data using pandas, return pandas dataframe
  dataset = pd.read_csv(filename)
  return dataset

def convert_labels(label):
  ## converts 'positive' and 'negative' to 1 and 0
  if label == 'positive':
    return 1
  elif label == 'negative':
    return 0

def separate_classes(X, y):
  ## for naive bayes
  ## separates samples into two classes (positive and negative)
  ## this exists because the sparse BOW encoding from CountVectorizer() is difficult
  ## to separate after the fact (converting to dense matrix takes up too much memory)
  samples_class_0 = []
  samples_class_1 = []
  for row in range(X.shape[0]):
    if y[row] == 0:
      samples_class_0.append(X[row])
    elif y[row] == 1:
      samples_class_1.append(X[row])
  return samples_class_0, samples_class_1

def extract_features(X_train, X_test, X_0, X_1):
    """Implement this for Part 1: Question 2"""
    count = CountVectorizer(encoding = 'utf-8', strip_accents='unicode', ngram_range=(1,2), stop_words='english', max_features=300)
    ## transforms data into sparse matrix
    bow_train = count.fit_transform(X_train)
    bow_test = count.transform(X_test)
    ## for naive bayes
    ## transforms the split dataset from separate_classes()
    bow_train_0 = count.transform(X_0)
    bow_train_1 = count.transform(X_1)
    return bow_train, bow_test, bow_train_0, bow_train_1

def splitIMDB(df):
  ## convert labels
  df['sentiment'] = df['sentiment'].apply(convert_labels)
  ## separate into review / sentiment
  X = df.review
  y = df.sentiment
  ## convert from pandas to numpy
  X_mat = X.to_numpy()
  y_mat = y.to_numpy()
  ## split into train / test
  X_train, X_test, y_train, y_test = train_test_split(X_mat, y_mat, test_size=.2)
  ## for naive bayes
  ## separate train data into classes
  X_train_0, X_train_1 = separate_classes(X_train, y_train)
  ## extract features, convert to BOW encoding
  X_train_f, X_test_f, X_train_0_f, X_train_1_f = extract_features(X_train, X_test, X_train_0, X_train_1)
  return (X_train_f,y_train), (X_test_f, y_test), X_train_0_f, X_train_1_f

####################

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0
 
if __name__ == '__main__':

    ## uncomment desired data file
    filename = 'disease.csv'
    # filename = 'IMDB_Dataset.csv'

    if filename == 'disease.csv':
      splitratio = 0.8
      dataset = loadcsv(filename)
      trainset, testset = splitdataset(dataset, splitratio)
      print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows.')
      hn_list = [64] #[8,16,32,64,128,256]
      acc_list = []
      
      for hidden_neurons in hn_list:
        params_NN = {'ni':trainset[0].shape[1], 'nh': hidden_neurons, 'no': 2}
        classalgs = {
                    # 'Random': algs.Classifier(),
                    # 'Naive Bayes': algs.NaiveBayes('disease'),
                    # 'Logistic Regression': algs.LogitReg(dataset='disease',learning_rate =.01, num_iterations = 100, run_stochastic=False),
                    'Neural Network': algs.NeuralNet(dataset='disease', params = params_NN, learning_rate = 0.01, num_iterations = 50, batch_size = 8, lambda_reg = 0.001)
                    }
        for learnername, learner in classalgs.items():
          print('Running learner = ' + learnername)
          # Train model
          learner.learn(trainset[0], trainset[1])
          predictions = learner.predict(testset[0])
          accuracy = getaccuracy(testset[1], predictions)
          print('Accuracy for ' + learnername + ': ' + str(accuracy))
          acc_list.append(accuracy)
      plt.plot(hn_list, acc_list)
      plt.xlabel("Number of hidden neurons")
      plt.ylabel("Accuracy")
      plt.savefig("Disease.png")

    elif filename == 'IMDB_Dataset.csv':
      dataset = loadIMDB(filename)
      trainset, testset, class_0, class_1 = splitIMDB(dataset)
      print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows')
      hn_list = [256]
      acc_list = []
      for hidden_neurons in hn_list:
        params_NN = {'ni':trainset[0].shape[1], 'nh': hidden_neurons, 'no': 2}
        classalgs = {
                    # 'Random': algs.Classifier(),
                    # 'Naive Bayes': algs.NaiveBayes('IMDB', class_0, class_1),
                    # 'Logistic Regression': algs.LogitReg(dataset='IMDB', learning_rate=.01, num_iterations=10, run_stochastic=True),
                    'Neural Network': algs.NeuralNet(dataset='IMDB',  params = params_NN, learning_rate = 0.01, num_iterations = 150, batch_size = 8, lambda_reg = 0.001)
                    }
        for learnername, learner in classalgs.items():
          print('Running learner = ' + learnername)
          # Train model
          learner.learn(trainset[0], trainset[1])
          predictions = learner.predict(testset[0])
          accuracy = getaccuracy(testset[1], predictions)
          print('Accuracy for ' + learnername + ': ' + str(accuracy))
          acc_list.append(accuracy)
      plt.plot(hn_list, acc_list)
      plt.xlabel("Number of hidden neurons")
      plt.ylabel("Accuracy")
      plt.savefig("Disease.png")
