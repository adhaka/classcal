import numpy as np
from rcc import check_consistency_size, check_consistency_class_weights
from rcc.utils import get_bin_freqs
from typing import List, Callable
import math


def compute_bin_mean(lst: List):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def compute_bin_entropy(lst: List):
    if len(lst) == 0:
        return 0
    
    jitter=1e-12
    lst_entropy = [a*math.log(a+jitter) + (1.-a)*math.log(1.-a+jitter) for a in lst]
    entropy = -sum(lst_entropy) / (len(lst)*math.log(2.))
    return entropy


def compute_expected_calibration_error(
    y_preds_probs: np.ndarray,
    y_true: np.ndarray,
    binning_scheme: Callable,
    power: int = 2,
    num_bins: int = 10,
) -> float:
    """Get Expected Calibration Error(ECE)
    Args:
        y_pred_probs: 1D or 2D array of output probabilities.
        y_true: 1D array of ground truth labels.
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
        threshold: probability threshold for deciding.

    Returns:
        A single scalar which calculates the ECE.

    """
    if not check_consistency_size(y_preds_probs, y_true):
        raise ValueError("Size of probabilities does not match outcome array size")

    jitter = 1e-9
    num_points = y_true.size
    bins_intervals = binning_scheme(num_bins)
    # bin_data_pair = list(zip(y_preds_probs, y_true))
    bin_points_collection, num_bin_points, num_pos_per_bin = get_bin_freqs(
        y_preds_probs, y_true, bins_intervals
    )

    bin_fracs = [(a / num_points + jitter) for a in num_bin_points]
    # compute the average confidence for each bin
    bin_conf_mean = list(map(compute_bin_mean, bin_points_collection))
    # compute the average accuracy for each bin
    bin_acc_fraction = [
        a / (b + jitter) for a, b in zip(num_pos_per_bin, num_bin_points)
    ]
    bin_errors = [abs(a - b) ** power for a, b in zip(bin_conf_mean, bin_acc_fraction)]
    ece = np.dot(bin_errors, bin_fracs) ** (1.0 / power)
    return ece

def compute_maximum_calibration_error(
    y_preds_probs: np.ndarray,
    y_true: np.ndarray,
    binning_scheme: Callable,
    power: int = 2,
    num_bins: int = 10
) -> float:
    """Get Maximum Calibration Error(MCE)
    Args:
        y_pred_probs: 1D or 2D array of output probabilities.
        y_true: 1D array of ground truth labels.
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
        threshold: probability threshold for deciding.
    Returns:
        A single scalar which calculates the MCE
    
    """
    if not check_consistency_size(y_preds_probs, y_true):
        raise ValueError("Size of probabilities does not match outcome array size")

    jitter = 1e-9
    num_points = y_true.size
    bins_intervals = binning_scheme(num_bins)
    bin_points_collection, num_bin_points, num_pos_per_bin = get_bin_freqs(
        y_preds_probs, y_true, bins_intervals
    )

    bin_fracs = [(a / num_points + jitter) for a in num_bin_points]
    # compute the average confidence for each bin
    bin_conf_mean = list(map(compute_bin_mean, bin_points_collection))
    # compute the average accuracy for each bin
    bin_acc_fraction = [
        a / (b + jitter) for a, b in zip(num_pos_per_bin, num_bin_points)
    ]
    bin_errors = [abs(a - b) ** power for a, b in zip(bin_conf_mean, bin_acc_fraction)]
    # take the maximum calibration error from all the bins
    mce = max(bin_errors)
    return mce

def compute_uncertainty_calibration_error(
    y_preds_probs: np.ndarray,
    y_true: np.ndarray,
    binning_scheme: Callable,
    num_bins: int = 10
) -> float:
    """Get Uncertainty Calibration Error(UCE):
    Args:
        y_pred_probs: 1D or 2D array of output probabilities
        y_true: 1D array of ground truth labels
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
        threshold: probability threshold for deciding.
    Returns:
        A single scalar which calculates the UCE
    
    """

    if not check_consistency_size(y_preds_probs, y_true):
        raise ValueError("Size of probabilities does not match outcome array size")

    jitter = 1e-9
    num_points = y_true.size
    bins_intervals = binning_scheme(num_bins)
    bin_points_collection, num_bin_points, num_pos_per_bin = get_bin_freqs(
        y_preds_probs, y_true, bins_intervals
    )

    bin_fracs = [(a / num_points + jitter) for a in num_bin_points]
    # compute the average entropy for each bin
    bin_entropy_mean = list(map(compute_bin_entropy, bin_points_collection))
    # compute the average error for each bin
    bin_acc_fraction = [
        a / (b + jitter) for a, b in zip(num_pos_per_bin, num_bin_points)
    ]
    bin_error_fraction = [1. - a for a in bin_acc_fraction]
    bin_calibration_error = [abs(a-b) for a,b in zip(bin_error_fraction, bin_entropy_mean)]
    uce = np.dot(bin_calibration_error, bin_fracs)
    print(uce)
    return uce

