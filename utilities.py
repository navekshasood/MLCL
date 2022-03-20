import math
import numpy as np
from scipy.optimize import fmin
import scipy

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def sigmoid(X):
    """ Compute the sigmoid function """
    den = 1.0 + np.exp(-1.0 * X)
    d = 1.0 / den
    return d

def dsigmoid(X):
    return X*(1-X)

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
            
#def fmin_simple(loss, initparams):
#    """ Temporarily simply calls fmin in scipy, should be replaced by your own personally written method """                   
#
#    return fmin(loss, initparams)

def gradient_descent(lr, theta, X, y):
    ## gradient descent
    m = y.size
    theta = theta - lr*(1.0/m) * np.dot(np.transpose(X),(np.dot(X,theta)-y))
    return theta

def cross_entropy(y_pred, y):
  loss=-np.sum(y*np.log(y_pred))
#   print (y_pred, np.log(y_pred))
  return loss/float(y_pred.shape[0])

def grad_cross_entropy(y_pred, y):
    return y_pred - y
