import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

def weighted_mean(vals, weights):
    w_sum = weights.sum()
    result = (vals * weights).sum() / w_sum
    return result

def weighted_std(vals, weights):
    n_weights = weights.size
    w_sum = weights.sum()
    w_mean = weighted_mean(vals, weights)
    result = np.sqrt(n_weights * (weights * np.power(vals - w_mean, 2)).sum() /
                     ((n_weights - 1) * w_sum))
    return result

def weighted_mean_error(weights):
    w_sum = weights.sum()
    result = np.sqrt(1 / w_sum)
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
    maxima = argrelextrema(pdf, np.greater)[0]
    minima = argrelextrema(pdf, np.less)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        result = np.full(samples.shape, 1)
    else:
        result = np.where(samples > x[minima[0]], 1, 0)

    return result

def _label_cluster_type(samples, weights, kde_result):
    result = -1
    pdf, x = kde_result["pdf"], kde_result["x"]
    a, b = 1.7, 2.3
    high_cutoff, low_cutoff = -2.5 * np.log10(a), -2.5 * np.log10(b)
    sigma_cutoff = 2.5 * np.log10(a) / 5
    maxima = argrelextrema(pdf, np.greater)[0]
    minima = argrelextrema(pdf, np.less)[0]
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
            weighted_mu = weighted_mean(samples_baseline, weights_baseline)
            weighted_sigma = weighted_std(samples_baseline, weights_baseline)
            errs_bright = np.power(weights_bright, -2)
            delta = weighted_mu - samples_bright
            delta_sigma = errs_bright + weighted_sigma
            delta_significance = delta / delta_sigma
            tol1 = (weighted_mu + low_cutoff < samples_bright)
            tol2 = (samples_bright < weighted_mu + high_cutoff)
            condition1 = all(tol1 & tol2)
            condition2 = (weighted_sigma <= sigma_cutoff)
            condition3 = all(delta_significance >= 5)

            if condition1 and condition2 and condition3:
                result = 2

    return result

def _kde_cluster_label(df, mag_column, magerr_column, bandwidth):
    samples = df[mag_column].values
    weights = np.power(df[magerr_column].values, -2)
    kde_result = kde_pdf(samples, weights, bandwidth)
    cluster_type = _label_cluster_type(samples, weights, kde_result)

    if cluster_type == -1:
        cluster_membership = np.full(len(samples), -1)
    else:
        cluster_membership = label_cluster_membership(samples, kde_result)

    d_keys = list(df.columns) + ["cluster_label"]
    d_vals = [df[col].values for col in df.columns] + [cluster_membership]
    d = {k: v for k, v in zip(d_keys, d_vals)}
    result = pd.DataFrame(d)
    return result

def _cl_transform(s, bandwidth):
    samples = s.transform(lambda x: x[0]).values
    weights = np.power(s.transform(lambda x: x[1]).values, -2)
    try:
        kde_result = kde_pdf(samples, weights, bandwidth)
    except ValueError:
        print(s.index)
        raise ValueError
    cluster_type = _label_cluster_type(samples, weights, kde_result)

    if cluster_type == -1:
        result = np.full(len(samples), -1)
    else:
        result = label_cluster_membership(samples, kde_result)

    return result
    

def cluster_label_dataframe(df, mag_column="mag_auto", 
                            magerr_column="magerr_auto", bandwidth=0.1254):
    mx = pd.MultiIndex.from_frame(df[["objectid", "filter"]])
    vals = list(df[[mag_column, magerr_column]].values)
    s = pd.Series(vals, index=mx)
    g = s.groupby(by=["objectid", "filter"])
    cluster_labelled = g.transform(_cl_transform, bandwidth)
    d_keys = list(df.columns) + ["cluster_label"]
    d_vals = [df[col].values for col in df.columns] + [cluster_labelled.values]
    d = {k: v for k, v in zip(d_keys, d_vals)}
    result = pd.DataFrame(d)
    return result

def make_lensing_dataframe(df, time_column="mjd", exp_time_column="exptime"):
    df_grouped = df.groupby(by=["objectid"])
    s = df_grouped.apply(_lens_apply, time_column, exp_time_column)
    s.name = "lens_data"
    result = s.reset_index()
    result['t_start'] = result["lens_data"].transform(lambda x: x.get('t_start'))
    result['t_end'] = result["lens_data"].transform(lambda x: x.get('t_end'))
    result['filters'] = result["lens_data"].transform(lambda x: x.get('filters'))
    result.drop(columns="lens_data", inplace=True, axis=1)
    return result

def _lens_apply(df, time_column, exp_time_column):
    result = {"t_start": np.nan, "t_end": np.nan, "filters": np.nan}
    
    s_per_day = 86400
    condition1 = ~((df["cluster_label"] == -1).any())
    condition2 = (df["cluster_label"] == 0).any()

    if condition1 and condition2:
        df = df.sort_values(by=[time_column])
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

        if case1:
            idx0, idx1 = baseline_idxs[boundary_idxs[0]], baseline_idxs[boundary_idxs[0] + 1]
            t_start = df.iloc[idx0][time_column] + (df.iloc[idx0][exp_time_column] / s_per_day)
            t_end = df.iloc[idx1][time_column]
        elif case2:
            idx0, idx1 = -1, baseline_idxs[0]
            t_start = np.nan
            t_end = df.iloc[idx1][time_column]
        elif case3:
            idx0, idx1 = baseline_idxs[-1], n_total
            t_start = df.iloc[idx0][time_column] + (df.iloc[idx0][exp_time_column] / s_per_day)
            t_end = np.nan

        if case1 or case2 or case3:
            lensed_idxs = np.arange(idx0 + 1, idx1)
            filters = ''.join(sorted(df.iloc[lensed_idxs]["filter"]))
            result = {"t_start": t_start, "t_end": t_end, "filters": filters}

    return result

# Rename this function, it doesn't return a dict
def _make_lensing_dict(df, time_column, exp_time_column):
    result = pd.DataFrame(data={"t_start": np.nan, "t_end": np.nan, "filters": np.nan}, index=[0])
    
    s_per_day = 86400
    condition1 = ~((df["cluster_label"] == -1).any())
    condition2 = (df["cluster_label"] == 0).any()

    if condition1 and condition2:
        df = df.sort_values(by=[time_column])
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

        if case1:
            idx0, idx1 = baseline_idxs[boundary_idxs[0]], baseline_idxs[boundary_idxs[0] + 1]
            t_start = df.iloc[idx0][time_column] + (df.iloc[idx0][exp_time_column] / s_per_day)
            t_end = df.iloc[idx1][time_column]
        elif case2:
            idx0, idx1 = -1, baseline_idxs[0]
            t_start = np.nan
            t_end = df.iloc[idx1][time_column]
        elif case3:
            idx0, idx1 = baseline_idxs[-1], n_total
            t_start = df.iloc[idx0][time_column] + (df.iloc[idx0][exp_time_column] / s_per_day)
            t_end = np.nan

        if case1 or case2 or case3:
            lensed_idxs = np.arange(idx0 + 1, idx1)
            filters = ''.join(sorted(df.iloc[lensed_idxs]["filter"]))
            result = pd.DataFrame(data={"t_start": t_start, "t_end": t_end, "filters": filters}, index=[0])

    return result