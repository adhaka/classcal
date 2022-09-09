import os
import numpy as np

def vectorize(f, a, axis=-1):
  if a.ndim > 1:
    return np.apply_along_axis(f, axis, a)
  else:
    return f(a)


def binarize(
  y_probs: np.ndarray,
  threshold:float=0.5
  ) -> np.ndarray:
  """
  Converts output probabilities to predictions(0 or 1.) 
  with threshold 
  Args: 
    y_probs:np.ndarray, output probabilities
    threshold:float, threshold probability 
  
  Returns:
    y_preds: np.ndarray 

  """
  y_preds = y_probs >= threshold
  return y_preds

def _probs_and_log_probs(
  y_probs:np.ndarray
) -> np.ndarray:
"""
helper function returns probabilities and log probabilities for class 1 and class 0

"""
jitter =1e-12
y_probs_log = np.log(y_probs + jitter)
y_probs_log1p = np.log(1. + jitter - y_probs)
return y_probs, (1. - y_probs), y_probs_log, y_probs_log1p
