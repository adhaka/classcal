import numpy as np
import unittest

from rcc.metrics import accuracy, brier_score, f1_score, get_counters, compute_expected_calibration_error, compute_maximum_calibration_error, compute_uncertainty_calibration_error

from rcc.utils import generate_uniform_spaced_bins, get_bins_from_equally_sliced_data


class TestMetrics(unittest.TestCase):

    def test_brier_score(self):
        self.assertAlmostEqual(brier_score(np.array([0.7]), np.array([1.])), 0.09, places=7)
        self.assertAlmostEqual(brier_score(np.array([0.3]), np.array([0.])), 0.09, places=7)
        self.assertAlmostEqual(brier_score(np.array([1.0]), np.array([1.])), 0.0, places=7)
        self.assertAlmostEqual(brier_score(np.array([0.3]), np.array([1.])), 0.49, places=7)
        self.assertAlmostEqual(brier_score(np.array([0.5]), np.array([1.])), 0.25, places=7)
        self.assertAlmostEqual(brier_score(np.array([0.5]), np.array([0.])), 0.25, places=7)
        self.assertAlmostEqual(brier_score(np.array([0.5, 0.5]), np.array([0., 1.])), 0.25, places=7)
    
    def test_accuracy(self):
        self.assertAlmostEqual(accuracy(np.array([0.1, 0.7, 0.4, 0.3, 0.82]), np.array([0, 1, 0, 0, 1])), 1., places=7)
        self.assertAlmostEqual(accuracy(np.array([0.1, 0.7, 0.4, 0.3, 0.82]), 
        np.array([0, 1, 1, 0, 1]), threshold=0.39), 1., places=7)
        self.assertRaises(ValueError, lambda: accuracy(np.array([0.1, 0.7, 0.4, 0.3, 0.82]), np.array([0, 1, 0, 0])))
    
    def test_checks(self):
        self.assertRaises(ValueError, lambda:brier_score(np.array([0.5, 0.5]), np.array([0., 1., 1.])))
    
    def test_counters(self):
        self.assertAlmostEqual(get_counters(np.array([0.1, 0.7, 0.4, 0.3, 0.82, 0.24, 0.95]),
         np.array([0,1,0,0,1, 1, 0])), (2,3,1,1))

    def test_f1score(self):
        self.assertAlmostEqual(f1_score(np.array([0.1, 0.7, 0.4, 0.3, 0.82]), np.array([0, 1, 0,0,1])), 1., places=5)
    
    def test_ece_guo(self):
        self.assertAlmostEqual(compute_expected_calibration_error(np.array([0.1, 0.7, 0.4, 0.3, 0.82, 0.08, 0.93, 0.45, 0.36, 0.39]), 
        np.array([0,1,0,0,1, 1, 0, 0, 0, 1]), generate_uniform_spaced_bins, num_bins=5, power=1), 0.277, places=3)

    def test_mce_guo(self):
        self.assertAlmostEqual(compute_maximum_calibration_error(np.array([0.1, 0.7, 0.4, 0.3, 0.82, 0.24, 0.95, 0.45, 0.36, 0.39]), 
        np.array([0,1,0,0,1, 1, 0, 0, 0, 1]), generate_uniform_spaced_bins, num_bins=5, power=1), 0.425, places=3)
    
    def test_uce_laves(self):
        self.assertAlmostEqual(compute_uncertainty_calibration_error(np.array([0.1, 0.7, 0.4, 0.3, 0.82, 0.24, 0.95, 0.45, 0.36, 0.39]), 
        np.array([0,1,0,0,1, 1, 0, 0, 0, 1]), generate_uniform_spaced_bins, num_bins=5), 0.31, places=2)

