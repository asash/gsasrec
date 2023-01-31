from aprec.losses.loss import ListWiseLoss
import tensorflow as tf

class SoftmaxCrossEntropy(ListWiseLoss): #used by bert
    def __init__(self,  *args, **kwargs):
        super().__init__()
        self.__name__ = "SoftmaxCrossEntropy"
        self.less_is_better = True
    
    def calc_per_list(self, y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

