"""
This module supplies lens_filter, unstable_filter, unimodal filter, 
and their helper functions.
"""

import numpy as np
from numba import njit
from utils.helpers import get_bounding_idxs

FLUX_DOUBLE = -2.5 * np.log10(2)

@njit
def _weighted_std_err(weights):
    """Computes weighted standard error of weighted mean"""
    result = np.sqrt(1 / weights.sum())
    return result

def unstable_filter(df, **kwargs):
    """Returns True if the photometry is unstable (too many peaks in the KDE)"""
    label_column = kwargs.get("label_column", "cluster_label")
    result = (df[label_column] == -1).any()
    return result

def lens_filter(df, **kwargs):
    """
    df must be sorted by time
    kwargs and defaults are samples_per_filter=2, achromatic=True,
    unique_filters=2, factor_of_two=True, mag_column='mag_auto',
    magerr_column='magerr_auto', label_column="cluster_label",
    mag_column and magerr_column are passed to _check_factor
    """
    samples_per_filter = kwargs.get("samples_per_filter", 2)
    unique_filters = kwargs.get("unique_filters", 2)
    check_achromatic = kwargs.get("achromatic", True)
    check_factor_of_two = kwargs.get("factor_of_two", True)
    label_column= kwargs.get("label_column", "cluster_label")
    cl = df[label_column].values
    lensed_idxs = get_bounding_idxs(cl)
    n_windows = len(lensed_idxs)

    if n_windows > 0:

        if check_achromatic:
            achromatic = [_check_achromaticity(df, pair, unique_filters)
                          for pair in lensed_idxs]
        else:
            achromatic = np.full(n_windows, True)

        if check_factor_of_two:
            g = df.groupby(by="filter")
            factor_of_two = [_check_factor(df, g, pair, **kwargs)
                             for pair in lensed_idxs]
        else:
            factor_of_two = np.full(n_windows, True)

        enough_samples = [pair[1] - pair[0] > samples_per_filter for pair in lensed_idxs]
        result = all(achromatic) & all(factor_of_two) & all(enough_samples)
    else:
        result = False

    return result

def _check_factor(df, df_gb, idx_bounds, **kwargs):
    mag_column = kwargs.get("mag_column", "mag_auto")
    magerr_column = kwargs.get("magerr_column", "magerr_auto")
    label_column = kwargs.get("label_column", "cluster_label")
    l, u = idx_bounds
    idx_range = np.arange(l+1, u)
    g_idx = df_gb.indices
    filters = df["filter"].iloc[l+1: u].unique()
    results = np.full(len(filters), False)

    for i, f in enumerate(filters):
        group = df_gb.get_group(f)
        mask_bright = np.isin(df_gb.indices[f], idx_range)
        mask_baseline = group[label_column].values.astype(bool)
        samples = group[mag_column].values
        weights = group[magerr_column].values**-2
        results[i] = _factor_of_two(samples, weights, mask_bright, mask_baseline)

    result = results.all()
    return result

@njit
def _factor_of_two(samples, weights, mask_bright, mask_baseline):
    mu0 = np.average(samples[mask_bright], weights=weights[mask_bright])
    sig0 = _weighted_std_err(weights[mask_bright])
    mu1 = np.average(samples[mask_baseline], weights=weights[mask_baseline])
    sig1 = _weighted_std_err(weights[mask_baseline])
    mu_diff = mu1 - mu0
    sig_diff = sig0 + sig1
    lower_bound = -FLUX_DOUBLE - 5 * sig_diff
    upper_bound = -FLUX_DOUBLE + 5 * sig_diff
    within_bounds = lower_bound < mu_diff < upper_bound
    five_sigma = mu_diff / sig_diff > 5
    result = np.logical_and(within_bounds, five_sigma)
    return result

def _check_achromaticity(df, idx_bounds, unique_filters):
    l, u = idx_bounds
    result = df["filter"].iloc[l+1: u].unique().size > (unique_filters - 1)
    return result

def unimodal_filter(df, **kwargs):
    """Returns true if all(df[label_column] == 1) otherwise returns false."""
    label_column = kwargs.get("label_column", "cluster_label")
    result = (df[label_column].values.astype(bool)).all()
    return result

def lightcurve_classifier(lc, **params):

    if unstable_filter(lc, **params):
        result = "unstable"
    elif lens_filter(lc, **params):
        result = "background"
    elif unimodal_filter(lc, **params):
        result = "unimodal"
    else:
        result = "NA"

    return result
