from aprec.losses.loss import ListWiseLoss
import tensorflow as tf

#https://arxiv.org/abs/2205.09310
class LogitNormLoss(ListWiseLoss): #used by bert
    def __init__(self, temperature=1, *args, **kwargs):
        super().__init__()
        self.__name__ = "LogitNormLoss"
        self.less_is_better = True
        self.temperature = temperature
    
    def calc_per_list(self, y_true, y_pred):
        norms = tf.expand_dims(tf.norm(y_pred, axis=-1), -1)
        logit_norms = tf.math.divide_no_nan(y_pred, norms)/self.temperature
        return tf.nn.softmax_cross_entropy_with_logits(y_true, logit_norms)
        