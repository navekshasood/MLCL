import csv
import random
import math
from re import A
import numpy as np
import algorithms as algs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

def plot(accuracy, hidden, title):
  plt.figure()
  plt.plot(hidden, accuracy)
  plt.xlabel("Hidden Units")
  plt.ylabel("Accuracy")
  plt.savefig(f'experiment_{title}.png')

def normalize_dataset(trainset, testset):
  maximums = np.amax(trainset[0], axis = 0)
  minimums = np.amin(trainset[0], axis = 0)
  diff = maximums - minimums
  trainset_n = (trainset[0]-minimums)/diff
  testset_n = (testset[0]-minimums)/diff
  return (trainset_n,trainset[1]), (testset_n,testset[1])

def downsample_dataset(trainset):
  x = np.unique(trainset[1], return_counts = True)
  min_count = np.min(x[1])
  class_0 = np.where(trainset[1]==0)
  class_1 = np.where(trainset[1]==1)
  idx_0 = random.sample(list(class_0[0]), min_count)
  idx_1 = random.sample(list(class_1[0]), min_count)
  idx = idx_0 + idx_1
  new_set = (trainset[0][idx], trainset[1][idx])
  return new_set



if __name__ == '__main__':

    ## uncomment desired data file
    filename = 'disease.csv'
    filename = 'IMDB_Dataset.csv'

    if filename == 'disease.csv':
      splitratio = 0.8
      dataset = loadcsv(filename)
      trainset_u, testset_u = splitdataset(dataset, splitratio)
      trainset_n, testset_n = normalize_dataset(trainset_u, testset_u)
      trainset = downsample_dataset(trainset_n)
      testset = downsample_dataset(testset_n)
      print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows.')
      hn_list = [128] #[8,16,32,64,128,256]
      acc_list = []
      
      for hidden_neurons in hn_list:
        params_NN = {'ni':trainset[0].shape[1], 'nh': hidden_neurons, 'no': 2}
        classalgs = {
                    'Random': algs.Classifier(),
                    'Naive Bayes': algs.NaiveBayes('disease'),
                    'Logistic Regression': algs.LogitReg(dataset='disease',learning_rate =0.01, num_iterations = 10, run_stochastic=False),
                    'Neural Network': algs.NeuralNet(dataset='disease', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 0.001, lr_annealing = False, regularization = False),
                    'Neural Network with LR Annealling': algs.NeuralNet(dataset='disease', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 0.001, lr_annealing = True, regularization = False),
                    'Neural Network with Regularization': algs.NeuralNet(dataset='disease', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 0.001, lr_annealing = False, regularization = True),
                    'Neural Network with LR Annealling & Regularization': algs.NeuralNet(dataset='disease', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 0.001, lr_annealing = True, regularization = True)
                    }
        for learnername, learner in classalgs.items():
          print('Running learner = ' + learnername)
          # Train model
          learner.learn(trainset[0], trainset[1])
          predictions = learner.predict(testset[0])
          accuracy = getaccuracy(testset[1], predictions)
          print('Accuracy for ' + learnername + ': ' + str(accuracy))
          acc_list.append(accuracy)
      # plot(acc_list,hn_list,filename)
      

    elif filename == 'IMDB_Dataset.csv':
      dataset = loadIMDB(filename)
      trainset, testset, class_0, class_1 = splitIMDB(dataset)
      print(f'Split {len(dataset)} rows into train={trainset[0].shape[0]} and test={testset[0].shape[0]} rows')
      hn_list = [128] #[8, 16, 32, 64, 128, 256, 512, 1024]
      acc_list = []
      for hidden_neurons in hn_list:
        params_NN = {'ni':trainset[0].shape[1], 'nh': hidden_neurons, 'no': 2}
        classalgs = {
                    'Random': algs.Classifier(),
                    'Naive Bayes': algs.NaiveBayes('IMDB', class_0, class_1),
                    'Logistic Regression': algs.LogitReg(dataset='IMDB', learning_rate=0.01, num_iterations=10, run_stochastic=True),
                    'Neural Network': algs.NeuralNet(dataset='IMDB', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 0.001, lr_annealing = False, regularization = False),
                    'Neural Network with LR Annealling': algs.NeuralNet(dataset='IMDB', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 1e-9, lr_annealing = True, regularization = False),
                    'Neural Network with Regularization': algs.NeuralNet(dataset='IMDB', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 1e-9, lr_annealing = False, regularization = True),
                    'Neural Network with LR Annealling & Regularization': algs.NeuralNet(dataset='IMDB', params = params_NN, learning_rate = 0.01, num_iterations = 30, batch_size = 4, lambda_reg = 1e-9, lr_annealing = True, regularization = True)
                    }
        for learnername, learner in classalgs.items():
          print('Running learner = ' + learnername)
          # Train model
          learner.learn(trainset[0], trainset[1])
          predictions = learner.predict(testset[0])
          accuracy = getaccuracy(testset[1], predictions)
          print('Accuracy for ' + learnername + ': ' + str(accuracy))
          acc_list.append(accuracy)
      # plot(acc_list,hn_list,filename)

# References:
# 1. https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# 2. https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/
