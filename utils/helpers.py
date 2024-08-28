"""
This module provides helper functions which are shared amongst
more than one module in the same directory
"""
import numpy as np

def get_bounding_idxs(cluster_label_array):
    """This function finds time-contiguous sequences of measurements within
    a lightcurve which were labelled bright by the gaussian KDE method
    within the kde_labeling module and returns the bounding indexes 
    (that is, the index of the sample preceding the bright sequence
    and the index of the sample following the bright sequence)."""
    n_total = len(cluster_label_array)
    idxs = np.arange(n_total)
    t_start = [i for i in idxs[:-1]
               if cluster_label_array[i] == 1 and cluster_label_array[i+1] == 0]
    t_end = [i+1 for i in idxs[:-1]
             if cluster_label_array[i] == 0 and cluster_label_array[i+1] == 1]

    if cluster_label_array[0] == 0:
        t_start = np.concatenate(([-1], t_start))
    if cluster_label_array[-1] == 0:
        t_end = np.concatenate((t_end, [n_total]))

    result = np.column_stack([t_start, t_end]).astype(int)
    return result
