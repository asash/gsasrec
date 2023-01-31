from aprec.evaluation.metrics.metric import Metric
from scipy.special import softmax
import numpy as np

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

class Confidence(Metric):
    def __init__(self, activation):
        self.name = f"{activation}Confidence"
        if activation == 'Softmax':
            self.activation = softmax
        elif activation == 'Sigmoid':
            self.activation = sigmoid 
        else:
            raise Exception(f"unknown activation {activation}")
            
        
    def __call__(self, recommendations, actual_actions):
        if len(recommendations) == 0:
            return 0
        scores = np.array([rec[1] for rec in recommendations])
        return self.activation(scores)[0]