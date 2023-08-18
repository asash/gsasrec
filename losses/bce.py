from aprec.losses.loss import ListWiseLoss
import tensorflow as tf


class BCELoss(ListWiseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "BCE"
        self.less_is_better = True

    def calc_per_list(self, y_true_raw, y_pred):
        eps = tf.constant(1e-8, y_pred.dtype)
        y_true = tf.cast(y_true_raw, y_pred.dtype)
        is_target = tf.cast((y_true >= -eps), y_pred.dtype)
        trues = y_true*is_target
        pos = trues*tf.math.softplus(-y_pred) * is_target
        neg = (1.0 - trues)*tf.math.softplus(y_pred) * is_target
        num_targets = tf.reduce_sum(is_target, axis=1)
        ce_sum = tf.reduce_sum(pos + neg, axis=1)
        res_sum = tf.math.divide_no_nan(ce_sum, num_targets)
        return res_sum

    def __call__(self, y_true_raw, y_pred):
        y_true = tf.cast(y_true_raw, y_pred.dtype)
        eps = tf.constant(1e-8, y_pred.dtype)
        is_target = tf.cast((y_true >= -eps), y_pred.dtype)
        trues = y_true*is_target
        pos = trues*tf.math.softplus(-y_pred) * is_target
        neg = (1.0 - trues)*tf.math.softplus(y_pred) * is_target
        num_targets = tf.reduce_sum(is_target)
        ce_sum = tf.reduce_sum(pos + neg)
        res_sum = tf.math.divide_no_nan(ce_sum, num_targets)
        return res_sum