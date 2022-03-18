import numpy as np
import utilities as utils
from collections import Counter
from math import log10 as log

# http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
class Classifier:
    """
    Generic classifier interface; returns random classification
    """
    
    def __init__( self ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest
            
class NaiveBayes(Classifier):

    def __init__( self, dataset, class_0=None, class_1=None):
        """ Initializes parameters for a naive bayes model """

        self.dataset = dataset # name of dataset
        self.num_classes = 2 # number of classes (0 and 1)
        self.num_features = None # 0-7 for disease, vocab size for IMDB
        self.num_samples = None # number of samples (num of reviews for IMDB)
        self.weights = None # learned weights for each feature for each class
        self.P_y = None # P(y), probability of each label

        ## optional arguments, relevant for IMDB
        ## class-separated data
        if class_0 is not None:
          self.class_0 = class_0
        if class_1 is not None:
          self.class_1 = class_1
        
    def learn(self, Xtrain, ytrain):

        ## calculate probability for each class
        sum_y = np.zeros((self.num_classes)) # shape (2,)
        for label in range(self.num_classes):
          sum_y[label] = (ytrain == label).sum()
        self.P_y = sum_y / sum_y.sum()
        #print(f"P_y:\n {self.P_y}")

        ## create lookup dict for each value of each feature for each class
        self.num_features = Xtrain.shape[1]
        self.num_samples = Xtrain.shape[0]

        if self.dataset == 'disease':
          ## key is column number in disease.csv, value is a Counter()
          ## where Counter() key is the actual number, value is the frequency count
          ## {column : {value_for_that_sample : freq_count}}
          ## freq_count then converted to a probability
          class_0 = {k: Counter() for k in range(self.num_features)}
          class_1 = {k: Counter() for k in range(self.num_features)}
          for i in range(self.num_samples):
            for j in range(self.num_features):
              feature = Xtrain[i][j]
              if ytrain[i] == 0:
                class_0[j][feature] += 1
              elif ytrain[i] == 1:
                class_1[j][feature] += 1

          ## convert frequency counts to probability
          for feature_category in class_0: # columns in disease.csv (0-7)
            for feature_value in class_0[feature_category]: # iterate through feature values
              ## if feature in both classes, average out
              if feature_value in class_1[feature_category]:
                total = class_0[feature_category][feature_value] + class_1[feature_category][feature_value]
                class_0[feature_category][feature_value] /= total
                class_1[feature_category][feature_value] /= total
              ## if feature value only in one class, P(feature value|class) = 1.0
              else:
                class_0[feature_category][feature_value] = 1.0

          ## convert feature_values that only appear in other class 
          for feature_category in class_1:
            for feature_value in class_1[feature_category]:
              if feature_value not in class_0[feature_category]:
                class_1[feature_category][feature_value] = 1.0 

        elif self.dataset == 'IMDB':
          ## first key is redundant in IMDB.csv, value is a Counter()
          ## where Counter() key is each word in the vocabulary, value is the frequency count
          ## {0 : {word : freq_count}}
          ## freq_count then converted to a probability
          class_0 = {k: Counter() for k in range(1)}
          class_1 = {k: Counter() for k in range(1)}
          #print(f"type:{type(Xtrain[0])}") 

          ## sum columns of the sparse BOW matrix
          ## results in [vocab_size] array containing
          ## frequency counts for each word for each class
          freq_0 = np.asarray(self.class_0.sum(axis=0))[0]
          freq_1 = np.asarray(self.class_1.sum(axis=0))[0]

          ## add these frequency counts to class_0 and class_1 dicts
          for word_index in range(self.num_features):
            class_0[0][word_index] += freq_0[word_index]
            class_1[0][word_index] += freq_1[word_index]

          ## convert frequency counts to probabilities
          for word in class_0[0]:
            ## if feature in both classes, average out
            if word in class_1[0]:
              total = class_0[0][word] + class_1[0][word]
              class_0[0][word] /= total
              class_1[0][word] /= total
            ## if feature value only in one class, P(feature value|class) = 1.0
            else:
              class_0[0][word] = 1.0

          ## convert feature_values that only appear in other class 
          for word in class_1[0]:
              if word not in class_0[0]:
                class_1[0][word] = 1.0 
        
        ## create master dict of learned probabilities
        self.weights = {0 : class_0, 1 : class_1}

    def predict(self, Xtest):

      ## argmax_k( sum_i( log P(feature_i|class_k) + log P(class_k)))
      num_test_samples = Xtest.shape[0]
      predictions = []

      for i in range(num_test_samples): # iterate through rows
        if i % 1000 == 0:
          print(f"SAMPLE: {i}\n" + "-"*100)

        ## create array for storing summation values, to be used with argmax
        summation = np.zeros((self.num_classes))

        ## iterate through each feature, grab the learned probability for that feature
        ## very slow with IMDB because we are looping through every word in vocab for each sample
        for j in range(self.num_features):
          ## grab learned proability from master lookup dict
          if self.dataset == 'disease':
            probability_0 = self.weights[0][j][Xtest[i][j]]
            probability_1 = self.weights[1][j][Xtest[i][j]]

          elif self.dataset == 'IMDB':
            ## skip words that contain 0 frequency counts
            ## i.e, only words in that particular test sample
            ## will have freq counts greater than 1
            probability_0 = None
            probability_1 = None
            sample = Xtest[i].toarray()[0]
            ## if this word appears in the test sample
            ## grab that probability for each class
            if sample[j] > 0: 
              probability_0 = self.weights[0][0][j]
              probability_1 = self.weights[1][0][j]

          ## accomodate unseen values and values seen only in one class
          ## use logs to avoid underflow
          ## for unseen values, I use a prob of 0.5 for each class
          ## for values seen only in one class, I use 0.01 for the opposite class
          ## I am guessing there better ways of going about this

          ## for IMDB, if word not in test set, skip this iteration
          if probability_0 == probability_1 == None:
            continue

          ## unseen value, use prob of 0.5 for each class
          if probability_0 == probability_1 == 0.0:
            summation[0] += log(0.5)
            summation[1] += log(0.5)

          ## only appears in class 1, use 0.01 for class 0
          elif probability_0 == 0.0:
            summation[0] += log(0.01)
            summation[1] += log(probability_1)

          ## only appears in class 0, use 0.01 for class 1
          elif probability_1 == 0.0:
            summation[0] += log(probability_0)
            summation[1] += log(0.01)

          ## use probability from lookup table for both classes
          else:
            summation[0] += log(probability_0)
            summation[1] += log(probability_1)

        ## multiply by P(y) and take the argmax
        summation[0] += log(self.P_y[0])
        summation[1] += log(self.P_y[1])
        predictions.append(np.argmax(summation))
        #print(f"Predictions for sample {i}: {predictions[i]}")

      return predictions
    

class LogitReg(Classifier):

    def __init__( self ):
        self.weights = None
        self.learning_rate = .01
        self.num_iters = 100
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.zeros(Xtrain.shape[1],)
        #lossfcn = lambda w: self.logit_cost(w, Xtrain,ytrain)
        #self.weights = utils.fmin_simple(lossfcn, self.weights)

        ## runs gradient descent, I avoided putting the loop in logit_cost()
        ## gradient descent formula is same as for linear regression
        ## difference is that error is calculated differenty
        for i in range(0, self.num_iters):
          cost = self.logit_cost(self.weights, Xtrain, ytrain) # currently don't do anything with this
          self.weights = utils.gradient_descent(self.learning_rate, self.weights, Xtrain, ytrain)
        
    def predict(self, Xtest):
        probs = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(probs)  
        return ytest
 
    def logit_cost(self, theta,X,y): 
        tt = X.shape[0] # number of training examples
        theta = np.reshape(theta,(len(theta),1))
        m = y.size

        J = (1./tt) * (-np.transpose(y).dot(np.log(utils.sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-utils.sigmoid(X.dot(theta)))))
    	
      ## should the loop for gradient descent be in here instead of in learn()?

    	# When you write your own minimizers, you will also return a gradient here
        return J[0]#,grad
        

class NeuralNet(Classifier):
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this test
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        """ Your implementation for learning the weights """
        pass
            
    def predict(self,Xtest):
      pass
    
    def evaluate(self, inputs):
      pass

#Accuracy for Naive Bayes: 85.22999999999999