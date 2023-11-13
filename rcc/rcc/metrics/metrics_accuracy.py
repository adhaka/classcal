from gc import get_count
from typing import Dict, Tuple, Union
import numpy as np

from rcc import check_consistency_size, check_consistency_shape, check_consistency_class_weights
from rcc import binarize

def accuracy(
    y_preds_probs: np.ndarray,
    y_true: np.ndarray,
    class_weights: Dict = None,
    sample_weights: np.ndarray = None,
    threshold: float = 0.5,
) -> float:
    """Get accuracy.

    Args:
        y_pred_probs: 1D or 2D array of output probabilities.
        y_true: 1D array of ground truth labels.
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
        threshold: probability threshold for deciding.
    
    Returns:
        A single scalar which calculates the accuracy.
    """

    #assert y_preds_probs.shape[1] == y_true.size
    y_true_unique = np.unique(y_true)

    if class_weights is not None:
        check_consistency_class_weights(y_true, class_weights)

    if sample_weights is not None:
        check_consistency_size(y_true, sample_weights)

    if not check_consistency_size(y_preds_probs, y_true):
        raise ValueError('Size of probabilities does not match outcome array size')

    y_preds_binary = binarize(y_preds_probs, threshold)
    num_correct = np.array(y_preds_binary == y_true, int).sum()
    acc = num_correct / y_true.size
    return acc

def brier_score(
    y_preds_probs:np.ndarray,
    y_true:np.ndarray,
    class_weights: Dict = None,
    sample_weights: np.ndarray = None,
) -> float:
    """Get brier score.

    Args:
        y_pred_probs: 1D or 2D array of output probabilities.
        y_true: 1D array of ground truth labels.
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
    
    Returns:
        A single scalar which calculates the brier score.
    """

    #assert y_preds_probs.shape[1] == y_true.size
    if class_weights is not None:
        check_consistency_class_weights(y_true, class_weights)

    if sample_weights is not None:
        check_consistency_size(y_true, sample_weights)
    
    if not check_consistency_size(y_preds_probs, y_true):
        raise ValueError('Size of probabilities does not match outcome array size')
    if np.unique(y_true).size > 2:
        raise ValueError('Only Binary classification is supported')
    y_positive = np.array(y_true == 1, int)
    if class_weights is not None:
        brierscore = np.average((y_positive -y_preds_probs)**2, weights=sample_weights)
    else:
        brierscore = np.average((y_positive - y_preds_probs)**2)

    return brierscore

def get_counters(
    y_preds_probs:np.ndarray,
    y_true:np.ndarray,
    threshold:float = 0.5,
    class_weights: Dict = None,
    sample_weights: np.ndarray=None
 ) -> Tuple:
    """Get sufficient statistics from data.

    Args:
        y_pred_probs: 1D or 2D array of output probabilities.
        y_true: 1D array of ground truth labels.
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
        threshold: probability threshold for deciding.
    
    Returns:
        A tuple containing numbers of true positive, true negative,
        false positive and false negative.
    """

    if not check_consistency_size(y_preds_probs, y_true):
        raise ValueError('Size of probabilities does not match outcome array size')

    y_preds_binary= binarize(y_preds_probs, threshold)
    num_points = y_preds_binary.size


    diff = y_preds_binary - y_true
    false_n = diff[diff== -1].size
    false_p = diff[diff== 1].size
    #mask1 = np.array(y_true == y_preds_binary, int)
    true_p = np.sum(y_preds_binary[diff==0])
    true_n = num_points - (true_p + false_n + false_p)
    return true_p, true_n, false_p, false_n

def f1_score(
    y_preds_probs:np.ndarray,
    y_true:np.ndarray,
    class_weights: Dict=None,
    sample_weights: np.ndarray = None,
    threshold: float = 0.5,
    *args,
    **kwargs
) -> float:
    """Get f1 score.

    Args:
        y_pred_probs: 1D or 2D array of output probabilities.
        y_true: 1D array of ground truth labels.
        class_weights: Dictionary of class respective weights(optional).
        sample_weights: 1D array of sample weights.
        threshold: probability threshold for deciding.
    
    Returns:
        A single scalar which calculates the f1 score.
    """

    jitter = 1e-9

    #assert y_preds_probs.shape[1] == y_true.size
    if class_weights is not None:
        check_consistency_class_weights(y_true, class_weights)

    if sample_weights is not None:
        check_consistency_size(y_true, sample_weights)
    
    true_p, true_n, false_p, false_n = get_counters(y_preds_probs, y_true, threshold, *args, **kwargs)
    precision = true_p/(true_p + false_p + jitter)
    recall = true_p/(true_p + false_n)
    f1score= 2*precision*recall /(precision + recall)
    return f1score

def get_all_accuracy_metrics(
    y_preds_probs:np.ndarray,
    y_true:np.ndarray,
    *args,
    **kwargs
)-> Dict[str, float]:
    """Single function to get all accuracy related metrics.

    Args: 
        y_pred_probs: 1D or 2D array of predicted probabilities, for binary classification.
        they represent probability of being 1 over 0.
        y_true: 1D array of ground truth labels

    Returns:
        A dictionary with sensitivity, specificity, recall, 
        accuracy, balanced accuracy and MCC.
    """

    jitter = 1e-9
    true_p, true_n, false_p, false_n = get_counters(y_preds_probs, y_true, *args, **kwargs)


    precision = true_p/ (true_p + false_p + jitter)
    recall = true_p/(true_p +false_n + jitter)

    sensitivity = recall
    specificity = true_n/(true_n+false_p + jitter)
    true_pr = sensitivity
    true_nr = specificity
    accuracy= (true_p + true_n)/(true_p+true_n+false_n+false_p)
    balanced_accuracy = (true_pr + true_nr)/2.

    mcc = (true_p*true_n - false_p*false_n)/ \
    np.sqrt((true_p+false_p)(true_p+false_n)(true_n + false_p)(true_n + false_n))

    metrics = dict()
    metrics['sensitivity'] = sensitivity
    metrics['recall'] = recall
    metrics['specificity'] = specificity
    metrics['true_pr'] = true_pr
    metrics['true_nr'] = true_nr
    metrics['precision'] = precision
    metrics['mcc'] = mcc
    metrics['accuracy'] = accuracy
    metrics['balanced_accuracy'] = balanced_accuracy
    return metrics

