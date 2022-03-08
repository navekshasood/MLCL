import numpy as np
import utilities as utils

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
        
    # Uncomment and define, as currently simply does parent    
    #def learn(self, Xtrain, ytrain):
    
    #def predict(self, Xtest):
    
    
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
