from collections import defaultdict
from unicodedata import category
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

    def __init__( self ):
        """ Initializes parameters for a naive bayes model """
        self.num_classes = 2
        self.num_features = 8
        self.weights = None
        self.P_y = None
        
    # Uncomment and define, as currently simply does parent    

    def learn(self, Xtrain, ytrain):

        ## calculate probability for each class
        sum_y = np.zeros((self.num_classes)) # shape (2,)
        for label in range(self.num_classes):
          sum_y[label] = (ytrain == label).sum()
        self.P_y = sum_y / sum_y.sum()
        #print(f"P_y:\n {self.P_y}")

        ## create lookup dict for each value of each feature for each class
        class_0 = {k: Counter() for k in range(self.num_features)}
        class_1 = {k: Counter() for k in range(self.num_features)}

        num_samples = Xtrain.shape[0]
        for i in range(num_samples):
          for j in range(self.num_features):
            feature = Xtrain[i][j]
            if ytrain[i] == 0:
              class_0[j][feature] += 1
            elif ytrain[i] == 1:
              class_1[j][feature] += 1

        for feature_category in class_0: # 0-7
          for feature_value in class_0[feature_category]: # iterate through feature values
            ## if feature in both classes, average out
            if feature_value in class_1[feature_category]:
              total = class_0[feature_category][feature_value] + class_1[feature_category][feature_value]
              class_0[feature_category][feature_value] /= total
              class_1[feature_category][feature_value] /= total
            ## if feature value only in one class, P(feature value|class) = 1.0
            else:
              class_0[feature_category][feature_value] = 1.0

        for feature_category in class_1: # 0-7
          for feature_value in class_1[feature_category]: # iterate through feature values
            ## if feature value only in one class, P(feature value|class) = 1.0
            if feature_value not in class_0[feature_category]: # left-overs from previous loop
              class_1[feature_category][feature_value] = 1.0 
        
        ## dict of learned probabilities
        ## {class 1: feature 0-7: values, class 2: feature 0-7: values}
        self.weights = {0 : class_0, 1 : class_1}

    def predict(self, Xtest):
      
      ## argmax_k( sum_i( log P(feature_i|class_k) + log P(class_k)))
      predictions = []
      num_samples = Xtest.shape[0]
      for i in range(num_samples): # iterate through rows
        summation = np.zeros((self.num_classes))
        for j in range(self.num_features): # iterate through each feature
          ## grab the probability from the lookup table
          probability_0 = self.weights[0][j][Xtest[i][j]]
          probability_1 = self.weights[1][j][Xtest[i][j]]

          ## accomodate unseen values and values seen only in one class
          # use logs to avoid underflow
          # for unseen values, here I just used a prob of 0.5
          # for values seen only in one class, I used 0.01 for the other class
          # there are probably better ways of doing this
          if probability_0 == probability_1 == 0.0: # unseen
            summation[0] += log(0.5)
            summation[1] += log(0.5)

          elif probability_0 == 0.0: # if only appears in class 1
            summation[0] += log(0.01)
            summation[1] += log(probability_1)

          elif probability_1 == 0.0: # if only appears in class 0
            summation[0] += log(probability_0)
            summation[1] += log(0.01)

          else: # otherwise use the probability from the lookup table
            summation[0] += log(probability_0)
            summation[1] += log(probability_1)

        summation[0] += log(self.P_y[0])
        summation[1] += log(self.P_y[1])
        predictions.append(np.argmax(summation))

      return predictions
    

class LogitReg(Classifier):

    def __init__( self ):
        self.weights = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.zeros(Xtrain.shape[1],)
        lossfcn = lambda w: self.logit_cost(w, Xtrain,ytrain)
        self.weights = utils.fmin_simple(lossfcn, self.weights)
        
    def predict(self, Xtest):
        probs = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(probs)  
        return ytest
 
    def logit_cost(self, theta,X,y): 
        tt = X.shape[0] # number of training examples
        theta = np.reshape(theta,(len(theta),1))
    	
        J = (1./tt) * (-np.transpose(y).dot(np.log(utils.sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-utils.sigmoid(X.dot(theta)))))
    	
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
