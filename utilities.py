import math
import numpy as np
from scipy.optimize import fmin

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

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
            
def fmin_simple(loss, initparams):
    """ Temporarily simply calls fmin in scipy, should be replaced by your own personally written method """                   
    return fmin(loss, initparams);
