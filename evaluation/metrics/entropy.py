from aprec.evaluation.metrics.metric import Metric
from scipy.special import softmax
from scipy.stats import entropy
import numpy as np

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

class Entropy(Metric):
    def __init__(self, activation, k):
        self.name = f"{activation}Entropy@{k}"
        if activation == 'Softmax':
            self.activation = softmax
        elif activation == 'Sigmoid':
            self.activation = sigmoid 
        else:
            raise Exception(f"unknown activation {activation}")
        self.k = k
            
        
    def __call__(self, recommendations, actual_actions):
        if len(recommendations) == 0:
            return 0
        scores = self.activation(np.array([rec[1] for rec in recommendations[:self.k]]))
        scores = scores/np.sum(scores) #normalize, so that we can treat them as probs
        return entropy(scores, base=2) / len(scores)