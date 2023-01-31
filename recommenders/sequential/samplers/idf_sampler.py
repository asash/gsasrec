import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from aprec.recommenders.sequential.samplers.sampler import NegativesSampler
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters


class IDFSampler(NegativesSampler):
    def __init__(self, data_parameters: SequentialDataParameters, num_negatives: int) -> None:
        super().__init__(data_parameters, num_negatives) 
        weights = tf.random.uniform((self.data_parameters.num_items,), 0, 1) 
        self.reset_logits(weights)
       
    def reset_logits(self, weights):
        probs = weights/tf.reduce_sum(weights) 
        self.logits = tf.expand_dims(tf.math.log(probs/tf.reduce_sum(probs)), 0)
    
    def fit(self, train_users):
        print("fitting idf negatives sampler...")
        item_counts = np.zeros(self.data_parameters.num_items)
        for user_seq in tqdm.tqdm(train_users):
            for timestamp, item in user_seq:
                item_counts[item] += 1
        item_counts = tf.constant(item_counts, 'float32' )
        numerator = tf.expand_dims(tf.constant(len(train_users), 'float32'), 0)
        EPS = 1-9
        #for items with zero zero interactions the result is negative, so we take relu to make them 9
        inverted_counts = tf.nn.relu(tf.math.log(tf.math.divide_no_nan(numerator, item_counts) + EPS)) 
        self.reset_logits(inverted_counts)

    def __call__(self, masked_sequences, labels):
        negatives = tf.random.categorical(self.logits, self.data_parameters.batch_size*self.data_parameters.sequence_length*self.num_negatives) 
        negatives = tf.reshape(negatives, (self.data_parameters.batch_size, self.data_parameters.sequence_length, self.num_negatives))
        return negatives


