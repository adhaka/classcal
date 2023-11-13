
from typing import Union, List, Tuple
import numpy as np


## data structures inspired by https://github.com/p-lambda/verified_calibration/
Bins = List[float]
Data = List[Tuple[float, float]]
BinnedData= List[Data]

jitter = 1e-9

def binarize(
    y_preds: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    y_binary = y_preds >= threshold
    return y_binary


def generate_uniform_spaced_bins(
    num_bins: int=10
) -> List[float]:
    bins = []
    for i in range(1, num_bins+1):
        bins.append( i *1.0/num_bins )
    return bins

def generate_bins_from_midpoints(
    probs: List[float]
) -> List[float]:
    sorted_probs = sorted(np.unique(probs))
    bins = []
    for i,val in enumerate(sorted_probs):
        mid_point = (sorted_probs[i] + sorted_probs)/2
        bins.append(mid_point)
    bins.append(1.0)
    return bins

def get_bins_from_equally_sliced_data(
    probs: np.ndarray,
    num_bins:int =10
) -> List[float]:

    probs_sorted= sorted(probs)
    if probs_sorted.ndim > 1:
        num_points = probs.shape[1]
    else:
        num_points = probs.size

    if num_points < num_bins:
        num_bins=num_points
    
    binned_probs = np.array_split(probs_sorted, num_bins)

    bins_intervals = []
    for i in range(len(binned_probs)):
        mid_prob = (binned_probs[i][-1] + binned_probs[i+1][-1])/2.
        bins_intervals.append(mid_prob)

    bins_intervals = list(set(bins_intervals))
    bins_intervals.append(1.0)
    return bins_intervals

def get_bin_freqs(
    y_preds_probs:np.ndarray,
    y_true: np.ndarray,
    bins_intervals: List[float]
):
    #sorted_y_preds_probs = np.sort(y_preds_probs)
    sorted_y_preds_probs, sorted_y_true = zip(*sorted(zip(y_preds_probs, y_true)))
    num_points = y_preds_probs.size
    num_bins = len(bins_intervals)
    j=0
    num_points_per_bin = []
    num_pos_per_bin = []
    bin_points_collection = []
    for i, val in enumerate(bins_intervals):
        bin_points = []
        if i == 0:
            low_val = 0
            high_val = val
        elif i > 0 and i < num_bins:
            low_val = bins_intervals[i-1]
            high_val = bins_intervals[i]
        else:
            break
        
        #print(low_val, high_val)
        k=0
        pos = 0
        while j< num_points and sorted_y_preds_probs[j] >= low_val and sorted_y_preds_probs[j]< high_val:
            bin_points.append(sorted_y_preds_probs[j])
            if sorted_y_true[j] == 1:
                pos = pos +1
            j = j+1
            k=k+1
        num_points_per_bin.append(k)
        num_pos_per_bin.append(pos)
        bin_points_collection.append(bin_points)
    
    return bin_points_collection, num_points_per_bin, num_pos_per_bin


def diff_freq_probs(
    bin_data:Data
) -> float:
    bin_data= np.array(bin_data)
    mean_preds_probs = np.mean(bin_data[:,0])
    empirical_freq = np.mean(bin_data[:,1])
    return mean_preds_probs - empirical_freq



def convert_to_logits(
    probs: np.ndarray,
    jitter: float=1e-12
) -> np.ndarray:
    probs = np.clip(probs, jitter, 1.- jitter)
    logits = np.log(probs / (1-probs))
    return logits