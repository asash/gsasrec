import tensorflow as tf
from aprec.recommenders.sequential.samplers.sampler import NegativesSampler
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters


class RandomNegativesSampler(NegativesSampler):
    def __init__(self, data_parameters: SequentialDataParameters, num_negatives:int) -> None:
        super().__init__(data_parameters, num_negatives)

    def fit(self, training_sequences):
        pass

    def __call__(self, masked_sequences, labels):
        negatives = tf.random.uniform((self.data_parameters.batch_size,
                                       self.data_parameters.sequence_length,
                                       self.num_negatives), dtype='int64', maxval=self.data_parameters.num_items)
        return negatives
