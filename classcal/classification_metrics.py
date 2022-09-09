import numpy as np

from _utils import *

def accuracy(y_probs, y_true, threshold = 0.5):
  y_preds = np.zeros_like(y_probs)
  y_preds =  binarize(y_probs)
  correct = y_preds == y_true
  return np.sum(correct)/y_true.size
