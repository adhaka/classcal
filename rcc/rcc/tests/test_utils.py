import numpy as np
import unittest

from rcc.utils import binarize, generate_uniform_spaced_bins, get_bin_freqs


class TestUtilMethods(unittest.TestCase):
    def test_generate_uniform_bins(self):
        self.assertEqual(generate_uniform_spaced_bins(10), [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        )
        self.assertEqual(generate_uniform_spaced_bins(5), [0.2, 0.4, 0.6, 0.8, 1.0])
    
    def test_binarize(self):
        self.assertTrue(np.all(binarize(np.array([0.11, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]))==
         np.array([0., 0., 0., 0., 1, 1, 1, 1, 1, 1])))


    def test_get_bins_freqs(self):
        #print(get_bin_freqs(np.array([0.1, 0.7, 0.4, 0.3, 0.82, 0.08, 0.93]), generate_uniform_spaced_bins(5)))
        self.assertEqual(get_bin_freqs(np.array([0.1, 0.7, 0.4, 0.3, 0.82, 0.08, 0.93, 0.45, 0.36, 0.39]), np.array([0,0,1,0,1,0,1,0,0,1]),
         generate_uniform_spaced_bins(5)),
        ([[0.08, 0.1], [0.3, 0.36, 0.39], [0.4, 0.45], [0.7], [0.82, 0.93]], [2, 3, 2, 1, 2], [0, 1, 1,0, 2]))
        print(generate_uniform_spaced_bins(5))
        self.assertEqual(get_bin_freqs(np.array([0.1, 0.4, 0.3, 0.82, 0.08, 0.93, 0.45, 0.36, 0.39]), np.array([0,1,0,1,0,1,0,0,1]),
         generate_uniform_spaced_bins(5)),
        ([[0.08, 0.1], [0.3, 0.36, 0.39], [0.4, 0.45], [], [0.82, 0.93]], [2, 3, 2, 0, 2], [0, 1, 1, 1, 2]))


