"""
This module supplies lens_filter, unstable_filter, unimodal filter, 
and their helper functions.
"""

from collections import Counter
import numpy as np
from numba import njit
from numpy.linalg import inv
from microlensing.helpers import (
    get_bounding_idxs,
    weighted_std_err,
    weighted_std
)

FLUX_DOUBLE = -2.5 * np.log10(2)
#TAN2_AB_OPT = [1.391, 8.476e-1]
TAN2_AB_OPT = [2.634, 9.986e-1]

def tan2_curve(x, a, b):
    result = a * np.tan(b * np.pi * x / 2)**2
    return result

def unstable_filter(df, **kwargs):
    """Returns True if the photometry is unstable (too many peaks in the KDE)"""
    label_column = kwargs.get("label_column", "cluster_label")
    result = (df[label_column] == -1).any()
    return result

def lens_filter(df, **kwargs):
    """
    kwargs and defaults are min_per_filter=2
    n_filters_req=3, factor_of_two=True, mag_column='mag_auto',
    magerr_column='magerr_auto', label_column="cluster_label",
    mag_column and magerr_column are passed to _check_factor
    """
    min_per_filter = kwargs.get("min_per_filter", 2)
    n_filters_req = kwargs.get("n_filters_req", 3)
    check_factor_of_two = kwargs.get("factor_of_two", True)
    label_column= kwargs.get("label_column", "cluster_label")
    df = df.sort_values(by="mjd")
    cl = df[label_column].to_numpy()
    lensed_idxs = get_bounding_idxs(cl)
    n_windows = len(lensed_idxs)

    if n_windows > 0:
        achromatic = [
            _check_achromaticity(
                df["filter"].iloc[pair[0]+1: pair[1]].to_numpy().flatten(),
                n_filters_req,
                min_per_filter
            )
            for pair in lensed_idxs
        ]

        if check_factor_of_two:
            g = df.groupby(by="filter")
            factor_of_two = [
                _check_factor(df, g, pair, **kwargs)
                for pair in lensed_idxs
            ]
        else:
            factor_of_two = np.full(n_windows, True)

        result = all(achromatic) & all(factor_of_two)
    else:
        result = False

    return result

def _factor_of_two_tan2(samples, weights, mask_bright, mask_baseline):
    mean_bright = np.average(
        samples[mask_bright],
        weights=weights[mask_bright]
    )
    mean_baseline = np.average(
        samples[mask_baseline],
        weights=weights[mask_baseline]
    )

    if mask_bright.sum() > 1:
        sigma_bright = weighted_std(
            samples[mask_bright],
            weights[mask_bright]
        )
    else:
        sigma_bright = weighted_std_err(weights[mask_bright])

    if mask_baseline.sum() > 1:
        sigma_baseline = weighted_std(
            samples[mask_baseline],
            weights[mask_baseline]
        )
    else:
        sigma_baseline = weighted_std_err(weights[mask_baseline])

    delta_mag = mean_bright - mean_baseline
    sigma = np.sqrt(sigma_bright**2 + sigma_baseline**2)
    y_pred = tan2_curve(delta_mag + 2.5 * np.log10(2), *TAN2_AB_OPT)
    above_curve = (sigma > y_pred)
    in_bounds = ((delta_mag + 2.5 * np.log10(2))**2 < TAN2_AB_OPT[1]**-2)
    result = above_curve & in_bounds
    return result

def _check_factor(df, df_gb, idx_bounds, **kwargs):
    mag_column = kwargs.get("mag_column", "mag_auto")
    magerr_column = kwargs.get("magerr_column", "magerr_auto")
    label_column = kwargs.get("label_column", "cluster_label")
    l, u = idx_bounds
    idx_range = np.arange(l+1, u)
    filters = df["filter"].iloc[l+1: u].unique()
    results = np.full(len(filters), False)

    for i, f in enumerate(filters):
        group = df_gb.get_group(f)
        mask_bright = np.isin(df_gb.indices[f], idx_range)
        mask_baseline = (group[label_column] == 1).to_numpy()
        samples = group[mag_column].to_numpy()
        weights = group[magerr_column].to_numpy()**-2
        results[i] = _factor_of_two_tan2(
            samples,
            weights,
            mask_bright,
            mask_baseline
        )

    result = results.all()
    return result

@njit
def _factor_of_two(samples, weights, mask_bright, mask_baseline):
    mean_bright = np.average(
        samples[mask_bright],
        weights=weights[mask_bright]
    )
    std_err_bright = weighted_std_err(weights[mask_bright])
    mean_baseline = np.average(
        samples[mask_baseline],
        weights=weights[mask_baseline]
    )
    std_err_baseline = weighted_std_err(weights[mask_baseline])
    mean_difference = mean_bright - mean_baseline
    std_err_diff = np.sqrt(std_err_baseline**2 + std_err_bright**2)
    lower_bound = FLUX_DOUBLE - 5 * np.sqrt(std_err_diff)
    upper_bound = FLUX_DOUBLE + 5 * np.sqrt(std_err_diff)
    within_bounds = lower_bound < mean_difference < upper_bound
    five_sigma = -mean_difference / std_err_diff > 5
    result = np.logical_and(within_bounds, five_sigma)
    return result

def _check_achromaticity(vals, n_filters_req, min_per_filter):
    c = Counter(vals)
    result = ((len(c.keys()) >= n_filters_req) &
              (np.array(list(c.values())) >= min_per_filter).all())
    return result

def unimodal_filter(df, **kwargs):
    """Returns true if all(df[label_column] == 1) otherwise returns false."""
    label_column = kwargs.get("label_column", "cluster_label")
    result = (df[label_column] == 1).all()
    return result

def lightcurve_classifier(lc, **kwargs):

    if unstable_filter(lc, **kwargs):
        result = "unstable"
    elif lens_filter(lc, **kwargs):
        result = "background"
    elif unimodal_filter(lc, **kwargs):
        result = "unimodal"
    else:
        result = "NA"

    return result

def maha(u, v, c_inverse):
    delta = u - v
    result = np.einsum("jl,jk,kl->l", delta, c_inverse, delta)
    return result

def mahalanobis_distance_ellipse(df, label_col="cluster_label"):
    m = df[label_col].astype(bool)
    baseline_ab_data = df.loc[m, ["asemi", "bsemi"]].to_numpy().transpose()
    mean_baseline_ab = baseline_ab_data.mean(axis=1, keepdims=True)
    ab_data = (
        df[["asemi", "bsemi"]].to_numpy().transpose()
    )
    cov_matrix = np.cov(baseline_ab_data)
    result_data = np.sqrt(maha(ab_data, mean_baseline_ab, inv(cov_matrix)))
    result = df.assign(maha_distance=result_data)
    return result

def class_star_filter(df, label_col="cluster_label"):
    m = ~df[label_col].astype(bool)
    result = (df.loc[m, "class_star"] > 0.9).all()
    return result

def morphology_filter(df, label_col="cluster_label"):
    m = ~df[label_col].astype(bool)
    result = (df.loc[m, "maha_distance"] < 3).all()
    return result

def area_filter(df, label_col="cluster_label"):
    m = ~df[label_col].astype(bool)
    result = (df.loc[m, "footprint_area_z_score"]**2 < 3**2).all()
    return result

def good_photometry(df):
    result = (
        class_star_filter(df) &
        morphology_filter(df) &
        area_filter(df)
    )
    return result

def good_detection_filter(df, label_column="cluster_label"):
    first_pass = lens_filter(
        df,
        min_per_filter=1,
        n_filters_req=2,
        label_column=label_column
    )

    if first_pass:
        if good_photometry(df):
            result = True
        else:
            result = can_recover(df)
    else:
        result = False

    return result

def can_recover(df, label_column="cluster_label"):
    m = (
        ~df[label_column].astype(bool) &
        (
            (df["class_star"] < 0.9) |
            (df["maha_distance"] > 3) |
            (df["footprint_area_z_score"] > 3)
        )
    )
    result = lens_filter(
        df.loc[~m],
        min_per_filter=1,
        n_filters_req=2
    )
    return result

