import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from numba import njit

FLUX_DOUBLE = -2.5 * np.log10(2)

def kde_pdf(samples, weights, bandwidth):
    kde = gaussian_kde(samples, bw_method=1, weights=weights)
    kde.set_bandwidth(bandwidth / np.sqrt(kde.covariance[0, 0]))
    low = samples.min() - 1
    high = samples.max() + 1
    x = np.linspace(low, high, num=100)
    pdf = kde(x)
    result = {"pdf": pdf, "x": x}
    return result

@njit
def weighted_std_err(weights):
    result = np.sqrt(1 / weights.sum())
    return result

@njit
def label_cluster_membership(samples, n_modes, minima):

    if n_modes == -1:
        result = np.full(samples.shape, -1)
    elif n_modes == 1:
        result = np.full(samples.shape, 1)
    else:
        result = np.where(samples > minima, 1, 0)

    return result

def _is_unimodal_shifted_factor_of_two(samples, weights, bandwidth, x_min):
    mask_bright = samples < x_min
    mask_baseline = ~mask_bright
    samples[mask_bright] -= FLUX_DOUBLE
    kde_result = kde_pdf(samples, weights, bandwidth)  
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = find_peaks(pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        result = True
    else:
        result = False

    samples[mask_bright] += FLUX_DOUBLE
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

def cluster_label_dataframe(df, 
                            mag_column="mag_auto", 
                            magerr_column="magerr_auto", 
                            bandwidth=0.13):
    g = df[["objectid", "filter", mag_column, magerr_column]].groupby(by=["objectid", "filter"], sort=False)
    cluster_label = g.apply(_cl_apply, bandwidth)
    result = df.assign(cluster_label=cluster_label)
    return result

def _cl_apply(df, bandwidth):
    samples = df[df.columns[2]].values
    weights = np.power(df[df.columns[3]].values, -2)
    idxs = df.index
    modality_result = _label_modality(samples, weights, bandwidth)
    result = label_cluster_membership(samples, modality_result["modes"], modality_result["min"])
    return pd.DataFrame(data=result, index=idxs)

def unstable_filter(df):
    result = ~((df["cluster_label"] == -1).any())
    return result

def lens_filter(df, **kwargs):
    min_n_samples = kwargs.get("min_n_samples", 2)
    result = False
    cl = df["cluster_label"].values
    cl_bool = cl.astype(bool)
    n_zeros = cl.size - cl.sum()

    if n_zeros >= min_n_samples:
        result = _looks_lensed(df, cl_bool, **kwargs)

    return result

def _looks_lensed(df, cl_bool, **kwargs):
    achromatic = kwargs.get("achromatic", True)
    factor_of_two = kwargs.get("factor_of_two", True)
    
    achromatic_bool = True
    factor_bool = True

    if achromatic:
        achromatic_bool = _check_achromaticity(df, cl_bool)

    if factor_of_two:
        factor_bool = _check_factor(df, **kwargs)

    result = all([achromatic_bool, factor_bool])
    return result

def _check_factor(df, **kwargs):
    mag_column = kwargs.get("mag_column", "mag_auto")
    magerr_column = kwargs.get("magerr_column", "magerr_auto")
    g = df.groupby(by="filter")
    results = np.full(g.ngroups, False)

    for i, (_, group) in enumerate(g):
        mask_baseline = group["cluster_label"].values.astype(bool)

        if mask_baseline.all():
            results[i] = True
        else:
            samples = group[mag_column].values
            weights = group[magerr_column].values**-2
            results[i] = _factor_of_two(samples, weights, mask_baseline)

    result = results.all()
    return result

@njit
def _factor_of_two(samples, weights, mask):
    mu0 = np.average(samples[~mask], weights=weights[~mask])
    sig0 = weighted_std_err(weights[~mask])
    mu1 = np.average(samples[mask], weights=weights[mask])
    sig1 = weighted_std_err(weights[mask])
    mu_diff = mu1 - mu0
    sig_diff = sig0 + sig1
    lower_bound = -FLUX_DOUBLE - 5 * sig_diff
    upper_bound = -FLUX_DOUBLE + 5 * sig_diff
    within_bounds = lower_bound < mu_diff < upper_bound
    five_sigma = mu_diff / sig_diff > 5
    result = np.logical_and(within_bounds, five_sigma)
    return result
        
def _check_achromaticity(df, cl_bool):
    result = df.loc[~cl_bool, "filter"].unique().size > 1
    return result

def _check_time_contiguity(df):
    result = False
    cl = df["cluster_label"]
    n_bright = len(cl) - cl.sum()

    if n_bright > 0:
        mask_baseline = df["cluster_label"].values.astype(bool)    
        n_total = len(df)
        idxs = np.arange(n_total)
        baseline_idxs = idxs[mask_baseline]
        idx_diffs = np.diff(baseline_idxs)
        boundary_idxs = np.where(idx_diffs == (n_bright + 1))[0]
        case1 = len(boundary_idxs) == 1
        case2 = baseline_idxs[0] == n_bright
        case3 = (n_total - 1) - baseline_idxs[-1] == n_bright

        if any([case1, case2, case3]):
            result = True

    return result

def make_lensing_dataframe(df, time_column="mjd", exp_time_column="exptime"):
    """This function assumes the dataframe has been sorted by the time_column
    and filtered using lens_filter"""
    column_list = ["objectid", time_column, exp_time_column, "cluster_label", "filter"]
    df_grouped = df[column_list].groupby(by=["objectid"], sort=False)
    result = df_grouped.apply(_lens_apply)
    return result

def _get_bounding_idxs(cluster_label_array):
    n_total = len(cluster_label_array)
    idxs = np.arange(n_total)
    t_start = [i for i in idxs[:-1] if cluster_label_array[i] == 1 and cluster_label_array[i+1] == 0]    
    t_end = [i+1 for i in idxs[:-1]if cluster_label_array[i] == 0 and cluster_label_array[i+1] == 1]

    if cluster_label_array[0] == 0:
        t_start = np.concatenate(([-1], t_start))
    if cluster_label_array[-1] == 0:
        t_end = np.concatenate((t_end, [n_total]))

    result = np.column_stack([t_start, t_end]).astype(int)
    return result

def _lens_apply(df):
    s_per_day = 86400
    cl_array = df["cluster_label"].values
    df_indices = df.index
    n_samples = len(cl_array)
    bounding_idxs = _get_bounding_idxs(cl_array)
    t_start_idxs = bounding_idxs[:, 0]
    t_end_idxs = bounding_idxs[:, 1]
    t_start_min = df.iloc[t_start_idxs + 1, 1].values
    t_end_min = (df.iloc[t_end_idxs - 1, 1] + df.iloc[t_end_idxs - 1, 2] / s_per_day).values

    if t_start_idxs[0] == -1:
        t_start0 = -np.inf
        t_start_max = np.concatenate([[t_start0], (df.iloc[t_start_idxs[1:], 1] + 
                                              (df.iloc[t_start_idxs[1:], 2] / s_per_day)).values])
    else:
        t_start_max = (df.iloc[t_start_idxs, 1] + 
                   (df.iloc[t_start_idxs, 2] / s_per_day)).values

    if t_end_idxs[-1] == n_samples:
        t_end0 = np.inf
        t_end_max = np.concatenate([df.iloc[t_end_idxs[:-1], 1].values, [t_end0]])
    else:
        t_end_max = df.iloc[t_end_idxs, 1].values

    filters = [''.join(df.loc[df_indices[idx_pair[0] + 1: idx_pair[1]], "filter"]) for idx_pair in bounding_idxs]
    data = {"t_start_max": t_start_max, 
            "t_end_max": t_end_max, 
            "t_start_min": t_start_min, 
            "t_end_min": t_end_min, 
            "filters": filters}
    result = pd.DataFrame(data=data)
    return result

def subtract_baseline(df, mag_column="mag_auto", magerr_column="magerr_auto"):
    df_grouped = df.groupby(by=["objectid", "filter"], group_keys=False, sort=False)
    result = df_grouped.apply(_subtract_baseline_apply, mag_column, magerr_column)
    return result

def _subtract_baseline_apply(df, mag_column, magerr_column):
    mask_baseline = df["cluster_label"].values.astype(bool)
    samples_baseline = df.loc[mask_baseline, mag_column].values
    weights_baseline = df.loc[mask_baseline, magerr_column].values**-2
    try:
        baseline = np.average(samples_baseline, weights=weights_baseline)
    except ZeroDivisionError:
        print(df["cluster_label"].values)
    baseline_err = np.sqrt(1 / weights_baseline.sum())
    result = df.assign(delta_mag=df[mag_column] - baseline,
                       delta_mag_err=df[magerr_column] + baseline_err)
    return result

@njit
def _find_t_next_other_filter(i, mjds, filters):
    this_filter = filters[i]
    result = np.inf

    for j in range(i + 1, len(mjds)):

        if filters[j] != this_filter:
            result = mjds[j]
            break

    return result

@njit
def _measure_time(t_this, t_floor, t_next_other_filter, taus):
    t = np.maximum(t_floor, t_next_other_filter)
    result = np.maximum(t_this + taus - t, 0)
    return result

@njit
def _compute_t_floor(t_this, taus, t_next_other_filter):
    result = np.maximum(t_this + taus, t_next_other_filter)
    return result

def measure_time(df, taus, filter_column="filter", mjd_column="mjd"):
    filters = df[filter_column].values.astype("U1")
    mjds = df[mjd_column].values
    result = _measure_total_time(taus, filters, mjds)
    return result

@njit
def _measure_total_time(taus, filters, mjds):
    t_floor = np.zeros(taus.shape)
    result = np.zeros(taus.shape)

    for i in range(len(mjds) - 1):
        t_this = mjds[i]
        t_next_other_filter = _find_t_next_other_filter(i, mjds, filters)

        if np.isfinite(t_next_other_filter):
            result += _measure_time(t_this, t_floor, t_next_other_filter, taus)
        else:
            break

        t_floor = _compute_t_floor(t_this, taus, t_next_other_filter)

    result -= _subtract_time(taus, mjds)

    return result
@njit
def _subtract_time(taus, mjds):
    result = np.maximum(taus - (mjds[-1] - mjds[0]), 0)
    return result

def unimodal_filter(df):
    result = (df["cluster_label"].values.astype(bool)).all()
    return result