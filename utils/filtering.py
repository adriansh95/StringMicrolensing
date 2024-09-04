"""
This module supplies lens_filter, unstable_filter, unimodal filter, 
and their helper functions.
"""

import numpy as np
from numba import njit
from .helpers import get_bounding_idxs

FLUX_DOUBLE = -2.5 * np.log10(2)

@njit
def _weighted_std_err(weights):
    """Computes weighted standard error of weighted mean"""
    result = np.sqrt(1 / weights.sum())
    return result

def unstable_filter(df):
    """Returns True if the photometry is unstable (too many peaks in the KDE)"""
    result = ~((df["cluster_label"] == -1).any())
    return result

def lens_filter(df, **kwargs):
    """
    df must be sorted by time.
    kwargs and defaults are min_n_samples=2, achromatic=True, 
    factor_of_two=True, mag_column='mag_auto', magerr_column='magerr_auto',
    mag_column and magerr_column are passed to _check_factor
    """
    min_n_samples = kwargs.get("min_n_samples", 2)
    check_achromatic = kwargs.get("achromatic", True)
    check_factor_of_two = kwargs.get("factor_of_two", True)
    cl = df["cluster_label"].values
    lensed_idxs = get_bounding_idxs(cl)
    n_windows = len(lensed_idxs)

    if n_windows > 0:
        df_index = df.index

        if check_achromatic:
            achromatic = [_check_achromaticity(df, df_index[pair[0]+1: pair[1]])
                          for pair in lensed_idxs]
        else:
            achromatic = np.full(n_windows, True)

        if check_factor_of_two:
            g = df.groupby(by="filter", sort=False)
            factor_of_two = [_check_factor(df, g, df_index[pair[0]+1: pair[1]], **kwargs)
                             for pair in lensed_idxs]
        else:
            factor_of_two = np.full(n_windows, True)

        enough_samples = [pair[1] - pair[0] > min_n_samples for pair in lensed_idxs]
        result = all(achromatic) & all(factor_of_two) & all(enough_samples)
    else:
        result = False

    return result

def _check_factor(df, g, df_index_slice, **kwargs):
    mag_column = kwargs.get("mag_column", "mag_auto")
    magerr_column = kwargs.get("magerr_column", "magerr_auto")
    filters = df.loc[df_index_slice, "filter"]
    filters_unique = filters.unique()
    results = np.full(len(filters), False)

    for i, f in enumerate(filters_unique):
        group = g.get_group(f)
        mask_baseline = group["cluster_label"].values.astype(bool)
        group_idxs = group.index
        mask_idxs = np.isin(group_idxs, df_index_slice)
        samples = group[mag_column].values
        weights = group[magerr_column].values**-2
        results[i] = _factor_of_two(samples, weights, mask_idxs, mask_baseline)

    result = results.all()
    return result

@njit
def _factor_of_two(samples, weights, mask_idxs, mask_baseline):
    mu0 = np.average(samples[mask_idxs], weights=weights[mask_idxs])
    sig0 = _weighted_std_err(weights[mask_idxs])
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

def _check_achromaticity(df, df_index_slice):
    result = df.loc[df_index_slice, "filter"].unique().size > 1
    return result


def unimodal_filter(df):
    """Returns true if all(df["cluster_label"] == 1) otherwise returns false."""
    result = (df["cluster_label"].values.astype(bool)).all()
    return result

def lightcurve_classifier(lc, **params):

    if ~unstable_filter(lc):
        result = "unstable"
    elif lens_filter(lc, **params):
        result = "background"
    elif unimodal_filter(lc):
        result = "unimodal"
    else:
        result = np.nan

    return result