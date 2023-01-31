import unittest
from aprec.losses.softmax_crossentropy import SoftmaxCrossEntropy
from transformers.modeling_tf_utils import TFCausalLanguageModelingLoss
import os
import tensorflow as tf

class SoftmaxCrossentropyLossTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def test_nll_loss0(self):
        y_true_sparse = tf.constant([[1]])
        y_true = [0, 1, 0, 0]
        y_pred = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        class HFTLossConfig(object):
            tf_legacy_loss = False

        hft_transformers_loss = TFCausalLanguageModelingLoss() 
        hft_transformers_loss.config = HFTLossConfig()
        hft_loss = hft_transformers_loss.hf_compute_loss(y_true_sparse, y_pred).numpy()[0]
        our_loss = SoftmaxCrossEntropy().calc_per_list(y_true, y_pred).numpy()[0] 
        self.assertEqual(hft_loss, our_loss)
        
        y_true_sparse = tf.constant([[1, -100, 2]])
        y_true = tf.constant([[0, 1, 0, 0], [-100, -100, -100, -100], [0, 0, 1, 0]]) 
        y_pred = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 1.0], [0.8, 0.0, 0.2, 0.0]])
        hft_loss = hft_transformers_loss.hf_compute_loss(y_true_sparse, y_pred)
        our_loss = SoftmaxCrossEntropy().loss_per_list(y_true, y_pred)
        self.assertEqual(our_loss, hft_loss)
        

if __name__ == "__main__":
    unittest.main()    