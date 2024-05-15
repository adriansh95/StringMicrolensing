import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

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

def label_cluster_membership(samples, kde_result):
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = find_peaks(pdf)[0]
    minima = find_peaks(-pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        result = np.full(samples.shape, 1)
    else:
        result = np.where(samples > x[minima[0]], 1, 0)

    return result

def _label_cluster_type(samples, weights, kde_result, tolerance):
    result = -1
    pdf, x = kde_result["pdf"], kde_result["x"]
    a, b = tolerance
    high_cutoff, low_cutoff = -2.5 * np.log10(a), -2.5 * np.log10(b)
    sigma_cutoff = 2.5 * np.log10(1.7) / 5
    maxima = find_peaks(pdf)[0]
    minima = find_peaks(-pdf)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        weighted_sigma = weighted_std(samples, weights)

        if weighted_sigma <= sigma_cutoff:
            result = 1

    elif n_maxima == 2:
        mask_bright = samples < x[minima[0]]
        mask_baseline = samples > x[minima[0]]
        samples_bright = samples[mask_bright]
        weights_bright = weights[mask_bright]
        samples_baseline = samples[mask_baseline]
        weights_baseline = weights[mask_baseline]
        n_bright, n_baseline = len(samples_bright), len(samples_baseline)

        if (n_baseline > 2 * n_bright) and (n_bright >= 2):
            weighted_mu = np.average(samples_baseline, weights=weights_baseline)
            weighted_sigma = weighted_std(samples_baseline, weights_baseline)
            errs_bright = np.power(weights_bright, -2)
            delta = weighted_mu - samples_bright
            delta_sigma = errs_bright + weighted_sigma
            delta_significance = delta / delta_sigma
            tol1 = (weighted_mu + low_cutoff < samples_bright)
            tol2 = (samples_bright < weighted_mu + high_cutoff)
            condition1 = all(tol1 & tol2)
            condition2 = (weighted_sigma <= sigma_cutoff)
#             condition3 = all(delta_significance >= 5)
            condition3 = all(delta_significance >= 3) #DELETE ME

            if condition1 and condition2 and condition3:
                result = 2

    return result

def cluster_label_dataframe(df, 
                            mag_column="mag_auto", 
                            magerr_column="magerr_auto", 
                            bandwidth=0.1254,
                            tolerance=(1.7, 2.3)):
    g = df[["objectid", "filter", mag_column, magerr_column]].groupby(by=["objectid", "filter"])
    cluster_label = g.apply(_cl_apply, bandwidth, tolerance)
    result = df.assign(cluster_label=cluster_label)
    return result

def _cl_apply(df, bandwidth, tolerance):
    samples = df[df.columns[2]].values
    weights = np.power(df[df.columns[3]].values, -2)
    idxs = df.index
    kde_result = kde_pdf(samples, weights, bandwidth)
    cluster_type = _label_cluster_type(samples, weights, kde_result, tolerance)

    if cluster_type == -1:
        result = np.full(len(samples), -1)
    else:
        result = label_cluster_membership(samples, kde_result)

    return pd.DataFrame(data=result, index=idxs)

def lens_filter(df):
    result = False
    condition1 = ~((df["cluster_label"] == -1).any())
    condition2 = (df["cluster_label"] == 0).any()

    if condition1 and condition2:
        mask_baseline = df["cluster_label"] == 1    
        n_total = len(df)
        n_bright = len(df[~mask_baseline])
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