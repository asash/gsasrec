import unittest
import numpy as np

class TestSinEmbedding(unittest.TestCase):
    def test_embedding(self):
        from aprec.recommenders.sequential.models.sasrec.sasrec import ExpPositionEncoding, SinePositionEncoding
        sinEncoder = SinePositionEncoding(50, 64)
        input = np.array([[0, 1, 2, 3],[1,2,3,4]])
        encoded = sinEncoder(input)
        self.assertEqual(encoded.shape, (2, 4, 64))

        expEncoder = ExpPositionEncoding(50, 64)
        input = np.array([[0, 1, 2, 3],[1,2,3,4]])
        encoded = expEncoder(input)
        self.assertEqual(encoded.shape, (2, 4, 64))

if __name__== "__main__":
    unittest.main()