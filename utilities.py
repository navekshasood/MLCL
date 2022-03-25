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
    den = 1.0 + np.exp(-1.0 * X)
    d = 1.0 / den
    return d*(1-d)

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
            
#def fmin_simple(loss, initparams):
#    """ Temporarily simply calls fmin in scipy, should be replaced by your own personally written method """                   
#
#    return fmin(loss, initparams)

def gradient_descent(lr, theta, b, X, y):
    ## gradient descent
    m = y.size
    output = sigmoid((np.dot(X,theta)+b))
    error = output-y
    dw = (1.0/m)*np.dot(np.transpose(X),error)
    theta -= lr*dw
    b -= lr*(1.0/m)*np.sum(error)
    return theta, b

def cross_entropy(y_pred, y):
  loss=-np.sum(y*np.log(y_pred + 1e-8))
  return loss

def grad_cross_entropy(y_pred, y):
    return y_pred - y
