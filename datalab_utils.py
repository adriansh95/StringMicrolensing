import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

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

def label_cluster_membership(samples, cluster_type):
    cl_type = cluster_type["type"]

    if cl_type == -1:
        result = np.full(samples.shape, -1)
    elif cl_type == 1:
        result = np.full(samples.shape, 1)
    else:
        result = np.where(samples > cluster_type["min"], 1, 0)

    return result

def _label_cluster_type(samples, weights, kde_result):
    result = {"type": -1, "min": None}
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = find_peaks(pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        result = {"type": 1, "min": None}
    elif n_maxima == 2:
        minima = find_peaks(-pdf)[0]
        mask_bright = samples < x[minima[0]]
        mask_baseline = samples > x[minima[0]]
        more_bright = sum(mask_bright) > sum(mask_baseline)

        if more_bright:
            mask_use = mask_bright
            mask_compare = mask_baseline
            flux_double = -FLUX_DOUBLE
        else:
            mask_use = mask_baseline
            mask_compare = mask_bright
            flux_double = FLUX_DOUBLE

        samples_use = samples[mask_use]
        weights_use = weights[mask_use]
        samples_compare = samples[mask_compare]
        weights_compare = weights[mask_compare]
        weighted_mu = np.average(samples_use, weights=weights_use)
        weighted_sigma = weighted_std(samples_use, weights_use)
        errs_compare = np.power(weights_compare, -2)
        delta = np.abs(weighted_mu - samples_compare)
        delta_sigma = errs_compare + weighted_sigma
        delta_significance = delta / delta_sigma
        above_lower_bound = (weighted_mu + flux_double - 5 * weighted_sigma < samples_compare)
        below_upper_bound = (samples_compare < weighted_mu + flux_double + 5 * weighted_sigma)
        within_bounds = all(above_lower_bound & below_upper_bound)
        significant_difference = all(delta_significance >= 5)

        if within_bounds and significant_difference:
            result = {"type": 2, "min": x[minima[0]]}

    return result

def cluster_label_dataframe(df, 
                            mag_column="mag_auto", 
                            magerr_column="magerr_auto", 
                            bandwidth=0.11):
    g = df[["objectid", "filter", mag_column, magerr_column]].groupby(by=["objectid", "filter"])
    cluster_label = g.apply(_cl_apply, bandwidth)
    result = df.assign(cluster_label=cluster_label)
    return result

def _cl_apply(df, bandwidth):
    samples = df[df.columns[2]].values
    weights = np.power(df[df.columns[3]].values, -2)
    idxs = df.index
    kde_result = kde_pdf(samples, weights, bandwidth)
    cluster_type = _label_cluster_type(samples, weights, kde_result)
    result = label_cluster_membership(samples, cluster_type)
    return pd.DataFrame(data=result, index=idxs)

def lens_filter(df):
    result = False
    n_bright = sum(df["cluster_label"] == 0)
    condition1 = ~((df["cluster_label"] == -1).any())
    condition2 = n_bright > 1

    if condition1 and condition2:
        mask_baseline = df["cluster_label"] == 1    
        n_total = len(df)
        idxs = np.arange(n_total)
        baseline_idxs = idxs[mask_baseline]
        idx_diffs = np.diff(baseline_idxs)
        boundary_idxs = np.where(idx_diffs == (n_bright + 1))[0]
        case1 = len(boundary_idxs) == 1
        case2 = baseline_idxs[0] == n_bright
        case3 = (n_total - 1) - baseline_idxs[-1] == n_bright

        if case1 or case2 or case3:
            result = True

    return result

def make_lensing_dataframe(df, time_column="mjd", exp_time_column="exptime"):
    """This function assumes the dataframe has been sorted by the time_column
    and filtered using lens_filter"""
    column_list = ["objectid", time_column, exp_time_column, "cluster_label", "filter"]
    df_grouped = df[column_list].groupby(by=["objectid"])
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
    df_grouped = df.groupby(by=["objectid", "filter"], group_keys=False)
    result = df_grouped.apply(_subtract_baseline_apply, mag_column, magerr_column)
    return result

def _subtract_baseline_apply(df, mag_column, magerr_column):
    mask_baseline = df["cluster_label"] == 1
    samples_baseline = df.loc[mask_baseline, mag_column].values
    weights_baseline = df.loc[mask_baseline, magerr_column].values**-2
    baseline = np.average(samples_baseline, weights=weights_baseline)
    result = df.assign(delta_mag=df[mag_column] - baseline)
    return result