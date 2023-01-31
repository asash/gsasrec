from unittest import TestCase
import unittest
from aprec.datasets.datasets_register import DatasetsRegister


class TestNetflixDataset(TestCase):
    def test_netflix(self):
        dataset = DatasetsRegister()["netflix_fraction_0.001"]()
        self.assertEqual(len(dataset), 97030)

if __name__ == "__main__":
    unittest.main()    
