from unittest import TestCase
import numpy as np

from pyaudioclassification import feature_extraction, train, predict

class Test(TestCase):
    def test_feat_extract(self):
        features = feature_extraction('pyaudioclassification/tests/data/')
        loaded_features = np.load('pyaudioclassification/tests/saved/feat.npy')
        self.assertTrue(np.array_equal(features, loaded_features))
