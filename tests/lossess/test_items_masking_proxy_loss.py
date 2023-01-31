import unittest
import numpy as np

class TestItemsMaskingProxyLoss(unittest.TestCase):
    def test_items_masking_proxy_loss(self):
        from aprec.losses.bpr import BPRLoss
        from aprec.losses.bce import BCELoss
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
        proxy_loss = ItemsMaksingLossProxy(BCELoss(), 2, 4)
        proxy_loss.set_batch_size(2)
        proxy_loss.set_num_items(10)

        ytrue = np.array([
            [
                [-100, -100, -100],
                [1, 0, 0],
                [-100, -100, -100], 
                [1, 0, 0]
            ],
            [
                [1, 0, 0],
                [-100, -100, -100],
                [-100, -100, -100], 
                [-100, -100, -100]

            ]
        ])
        np.random.seed(31337)
        ypred = np.random.rand(2, 4, 3)
        result = proxy_loss(ytrue, ypred) 
        print(result)


if __name__ == "__main__":
    unittest.main()