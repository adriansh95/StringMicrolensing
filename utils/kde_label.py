"""
This module supplies the cluster_label_dataframe function and its
helper functions.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from numba import njit

def kde_pdf(samples, weights, bandwidth):
    """Makes a weighted gaussian kernel density estimation
    using the samples, weights, and specified bandwidth.
    Evaluates the KDE on 100 linearly spaced points (x) and
    returns a dict containing the x and y(x).
    """
    kde = gaussian_kde(samples, bw_method=1, weights=weights)
    kde.set_bandwidth(bandwidth / np.sqrt(kde.covariance[0, 0]))
    low = samples.min() - 1
    high = samples.max() + 1
    x = np.linspace(low, high, num=100)
    pdf = kde(x)
    result = {"pdf": pdf, "x": x}
    return result

def _label_modality(samples, weights, bandwidth):
    result = {"modes": -1, "min": np.nan}
    kde_result = kde_pdf(samples, weights, bandwidth)
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = find_peaks(pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        result = {"modes": 1, "min": np.nan}
    elif n_maxima == 2:
        minima = find_peaks(-pdf)[0]
        result = {"modes": 2, "min": x[minima[0]]}

    return result

def _cl_apply(df, bandwidth, mag_column, magerr_column):
    samples = df[mag_column].values
    weights = np.power(df[magerr_column].values, -2)
    idxs = df.index
    modality_result = _label_modality(samples, weights, bandwidth)
    result = _label_cluster_membership(samples, modality_result["modes"], modality_result["min"])
    return pd.DataFrame(data=result, index=idxs)

def cluster_label_dataframe(df,
                            mag_column="mag_auto",
                            magerr_column="magerr_auto",
                            bandwidth=0.13):
    """Groups the dataframe by objectid and filter, applies a gaussian
    KDE to the magnitudes using the specified bandwidth, and labels the
    cluster membership of each sample. 1 encodes baseline, 0 encodes bright
    excursions from baseline, and -1 encodes a star with unstable photometry
    (too many peaks in the KDE)."""
    g = df.groupby(by=["objectid", "filter"], sort=False, group_keys=False)
    cluster_label = g.apply(_cl_apply, bandwidth, mag_column, magerr_column)
    result = df.assign(cluster_label=cluster_label)
    return result

@njit
def _label_cluster_membership(samples, n_modes, minima):

    if n_modes == -1:
        result = np.full(samples.shape, -1)
    elif n_modes == 1:
        result = np.full(samples.shape, 1)
    else:
        result = np.where(samples > minima, 1, 0)

    return result
