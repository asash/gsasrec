import unittest
from aprec.losses.logit_norm import LogitNormLoss
import os
import tensorflow as tf

class LogitNormsTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def test_logitnorm_loss(self):
        loss = LogitNormLoss()
        y_true = tf.constant([[0, 1, 0, 0], [1, 0, 0, 0.]])
        y_pred = tf.constant([[0.2, -1, 5, 7], [2, 1, 1, 1.]])
        expected = [1.8970, 1.1170]
        result = loss.calc_per_list(y_true, y_pred).numpy()
        self.assertAlmostEqual(expected[0], result[0], places=3)
        self.assertAlmostEqual(expected[1], result[1], places=3)

        loss = LogitNormLoss(2)
        result = loss.calc_per_list(y_true, y_pred).numpy()
        self.assertAlmostEquals(result[1], 1.2480, 4)
        
if __name__ == "__main__":
    unittest.main()    