
import numpy as np
from typing import Dict

def check_consistency_size(
    y_target1: np.ndarray,
    y_target2: np.ndarray
)-> bool:
    
    return y_target1.size == y_target2.size


def check_consistency_shape(
    y_target1: np.ndarray,
    y_target2: np.ndarray
)-> bool:

    size11,size12 = y_target1.shape
    size21,size22 = y_target2.shape

    if size11 == size21 and size12 == size22:
        return True

    return False

def check_consistency_class_weights(
    y_target1: np.ndarray,
    class_weights:Dict
) -> bool:

    num_classes= np.unique(y_target1)
    if len(class_weights.keys()) == num_classes:
        return True
    
    return False
