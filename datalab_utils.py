import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from numba import njit

FLUX_DOUBLE = -2.5 * np.log10(2)

def weighted_std(vals, weights):
    result = np.sqrt(np.cov(vals, aweights=weights).item())
    return result

def kde_pdf(samples, weights, bandwidth):
    kde = gaussian_kde(samples, bw_method=1, weights=weights)
    kde.set_bandwidth(bandwidth / np.sqrt(kde.covariance[0, 0]))
    low = samples.min() - 1
    high = samples.max() + 1
    x = np.linspace(low, high, num=100)
    pdf = kde(x)
    result = {"pdf": pdf, "x": x}
    return result

def label_cluster_membership(samples, cluster_type, minima):

    if cluster_type == -1:
        result = np.full(samples.shape, -1)
    elif cluster_type == 1:
        result = np.full(samples.shape, 2)
    else:
        result = np.where(samples > minima, 1, 0)

    return result

def _is_cluster_type_2(samples, weights, bandwidth, x_min):
    mask_bright = samples < x_min
    mask_baseline = ~mask_bright
    samples[mask_bright] -= FLUX_DOUBLE
    kde_result = kde_pdf(samples, weights, bandwidth)  
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = find_peaks(pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        unimodal_shifted = True
    else:
        unimodal_shifted = False

    samples[mask_bright] += FLUX_DOUBLE
#     more_bright = sum(mask_bright) > sum(mask_baseline)

#     if more_bright:
#         mask_use = mask_bright
#         mask_compare = mask_baseline
#     else:
#         mask_use = mask_baseline
#         mask_compare = mask_bright

#     samples_use = samples[mask_use]
#     weights_use = weights[mask_use]
#     samples_compare = samples[mask_compare]
#     weights_compare = weights[mask_compare]
#     weighted_mu = np.average(samples_use, weights=weights_use)
#     weighted_sigma = weighted_std(samples_use, weights_use)
#     errs_compare = np.power(weights_compare, -2)
#     delta = np.abs(weighted_mu - samples_compare)
#     delta_sigma = errs_compare + weighted_sigma
#     delta_significance = delta / delta_sigma
#     significant_difference = (delta_significance >= 5).all()
#     result = unimodal_shifted and significant_difference
    result = unimodal_shifted
    return result

def _label_cluster_type(samples, weights, bandwidth):
    kde_result = kde_pdf(samples, weights, bandwidth)  
    result = {"type": -1, "min": np.nan}
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = find_peaks(pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        result = {"type": 1, "min": np.nan}
    elif n_maxima == 2:
        minima = find_peaks(-pdf)[0]

        if _is_cluster_type_2(samples, weights, bandwidth, x[minima[0]]):
            result = {"type": 2, "min": x[minima[0]]}

    return result

def cluster_label_dataframe(df, 
                            mag_column="mag_auto", 
                            magerr_column="magerr_auto", 
                            bandwidth=0.11):
    g = df[["objectid", "filter", mag_column, magerr_column]].groupby(by=["objectid", "filter"], sort=False)
    cluster_label = g.apply(_cl_apply, bandwidth)
    result = df.assign(cluster_label=cluster_label)
    return result

def _cl_apply(df, bandwidth):
    samples = df[df.columns[2]].values
    weights = np.power(df[df.columns[3]].values, -2)
    idxs = df.index
    cluster_type = _label_cluster_type(samples, weights, bandwidth)
    result = label_cluster_membership(samples, cluster_type["type"], cluster_type["min"])
    return pd.DataFrame(data=result, index=idxs)

def lens_filter(df):
    result = False
    cl = df["cluster_label"].values
    mask_twos = cl == 2
    n_zeros = sum(cl == 0)
    condition1 = ~((cl == -1).any())
    condition2 = n_zeros > 0

    if condition1 and condition2:
        
        if (cl == 2).any():
            looks_lensed = [False] * 2

            for i in range(2):
                df.loc[mask_twos, "cluster_label"] = i
                looks_lensed[i] = _looks_lensed(df)

            df["cluster_label"] = cl
            result = any(looks_lensed)

        else:
            result = _looks_lensed(df)

    return result

def _looks_lensed(df):
    result = False
    n_bright = sum(df["cluster_label"] == 0)

    if n_bright > 1:
        mask_baseline = df["cluster_label"] == 1    
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
    result.reset_index(level=1, inplace=True, drop=True)
    return result

def _lens_apply(df):
    s_per_day = 86400
    mask_bright = df["cluster_label"] == 0
    starts_lensed = mask_bright.iloc[0]
    ends_lensed = mask_bright.iloc[-1]
    n_lensed = sum(mask_bright)
    n_total = len(df)

    if starts_lensed:
        t_end_idx = n_lensed
        t_start = np.nan
        t_end = df.iat[t_end_idx, 1]
    elif ends_lensed:
        t_start_idx = -(n_lensed + 1)
        t_start = df.iat[t_start_idx, 1] + (df.iat[t_start_idx, 2] / s_per_day)
        t_end = np.nan
    else:
        idx_range = np.arange(n_total)
        t_start_idx = idx_range[mask_bright].min() - 1
        t_end_idx = idx_range[mask_bright].max() + 1
        t_start = df.iat[t_start_idx, 1] + (df.iat[t_start_idx, 2] / s_per_day)
        t_end = df.iat[t_end_idx, 1]

    filters = ''.join(df.loc[mask_bright, "filter"])
    result = pd.DataFrame(data={"t_start": [t_start], "t_end": [t_end], "filters": [filters]})
    return result

def subtract_baseline(df, mag_column="mag_auto", magerr_column="magerr_auto"):
    df_grouped = df.groupby(by=["objectid", "filter"], group_keys=False, sort=False)
    result = df_grouped.apply(_subtract_baseline_apply, mag_column, magerr_column)
    return result

def _subtract_baseline_apply(df, mag_column, magerr_column):
    mask_baseline = df["cluster_label"] == 1
    samples_baseline = df.loc[mask_baseline, mag_column].values
    weights_baseline = df.loc[mask_baseline, magerr_column].values**-2
    baseline = np.average(samples_baseline, weights=weights_baseline)
    result = df.assign(delta_mag=df[mag_column] - baseline)
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

def _measure_time_apply(df, taus, column_list):
    object_id = df.iloc[0, 0]
    filters = df[column_list[1]].values.astype("U1")
    mjds = df[column_list[2]].values
    t_floor = np.zeros(taus.shape)
    times = np.zeros(taus.shape)

    for i in range(len(df) - 1):
        t_this = mjds[i]
        t_next_other_filter = _find_t_next_other_filter(i, mjds, filters)

        if np.isfinite(t_next_other_filter):
            times += _measure_time(t_this, t_floor, t_next_other_filter, taus)
        else:
            break

        t_floor = _compute_t_floor(t_this, taus, t_next_other_filter)

    result = pd.DataFrame(data={"lensable_time": times}, index=taus)
    return result

def measure_lensable_time(df, taus, filter_column="filter", mjd_column="mjd"):
    column_list = ["objectid", filter_column, mjd_column]
    result = df[column_list].groupby(by="objectid", sort=False).apply(_measure_time_apply, taus, column_list)
    return result

def unimodal_filter(df):
    result = (df["cluster_label"] == 2).all()
    return result