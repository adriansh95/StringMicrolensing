import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

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
    kde = KernelDensity(bandwidth=bandwidth).fit(samples.reshape(-1, 1), sample_weight=weights)
    low = samples.min() - 1
    high = samples.max() + 1
    x = np.linspace(low, high, num=100)
    pdf = np.exp(kde.score_samples(x.reshape(-1, 1)))
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


def count_n_maxima(kde_result):
    pdf, x = kde_result["pdf"], kde_result["x"]
    maxima = argrelextrema(pdf, np.greater)[0]
    result = len(maxima)
    return result

def _label_cluster_bool(samples, weights, kde_result):
    result = False
    pdf, x = kde_result["pdf"], kde_result["x"]
    a, b = 1.7, 2.3
    high_cutoff, low_cutoff = -2.5 * np.log10(a), -2.5 * np.log10(b)
    sigma_cutoff = np.log10(a) / 2
    maxima = argrelextrema(pdf, np.greater)[0]
    minima = argrelextrema(pdf, np.less)[0]
    n_maxima = len(maxima)

    if n_maxima == 1:
        weighted_sigma = weighted_std(samples, weights)

        if weighted_sigma <= sigma_cutoff:
            result = True

    elif n_maxima == 2:
        mask_bright = samples < x[minima[0]]
        mask_baseline = samples > x[minima[0]]
        samples_bright = samples[mask_bright]
        weights_bright = weights[mask_bright]
        samples_baseline = samples[mask_baseline]
        weights_baseline = weights[mask_baseline]
        n_bright, n_baseline = len(samples_bright), len(samples_baseline)

        if n_baseline > 2 * n_bright:
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
                result = True

    return result

def cluster_label_dataframe(df, mag_column="mag_auto", magerr_column="magerr_auto",
                            time_column="mjd", kde_bandwidth=0.1254):
    filters = df["filter"].unique()
    n_filters = len(filters)
    label_cluster = [False for i in range(n_filters)]
    kdes = [{"a": np.zeros(100), "b": np.zeros(100)} for i in range(n_filters)]
    mask_filters = [df["filter"] == filters[i] for i in range(n_filters)]

    for i in range(n_filters):
        samples = df.loc[mask_filters[i], mag_column].values
        weights = np.power(df.loc[mask_filters[i], magerr_column].values, -2)
        kde_result = kde_pdf(samples, weights, kde_bandwidth)
        kdes[i] = kde_result
        label_cluster[i] = _label_cluster_bool(samples, weights, kde_result)

    if all(label_cluster):
        d_keys = list(df.columns) + ["cluster_label"]
        d_vals = [df[col].values for col in df.columns] + [np.full(len(df), -1)]
        d = {k: v for k, v in zip(d_keys, d_vals)}
        df_cluster_labelled = pd.DataFrame(d)

        for i in range(n_filters):
            m = df_cluster_labelled["filter"] == filters[i]
            samples = df_cluster_labelled.loc[m, mag_column].values
            df_cluster_labelled.loc[m, "cluster_label"] = label_cluster_membership(samples, kdes[i])

        df_cluster_labelled.sort_values(by=time_column, ignore_index=True, inplace=True)
        result = df_cluster_labelled
    else:
        result = None

    return result

def analyze_lensing_window(df, time_column="mjd", exp_time_column="exptime"):
    s_per_day = 86400
    mask_normal = df["cluster_label"] != 0
    normal_values_idxs = df.loc[mask_normal].index.to_numpy()
    idx_diffs = np.diff(normal_values_idxs)
    gap_idxs = np.where(idx_diffs > 1)[0]

    if len(gap_idxs) == 1:
        idx0, idx1 = normal_values_idxs[gap_idxs[0]], normal_values_idxs[gap_idxs[0] + 1]
        n_samples = (idx1 - idx0) - 1
        time_window = (df.at[idx0, time_column] + (df.at[idx0, exp_time_column] / s_per_day), df.at[idx1, time_column])
        window_idxs = np.arange(idx0 + 1, idx1)
        filters = ''.join(sorted(df.iloc[window_idxs]["filter"]))
        result = {"n_samples": n_samples, "time_window": time_window, "filters": filters}
    else:
        result = None

    return result

def detect_excursions(magnitudes_array, errors_array, kde_bandwidth=0.1254, 
                      tolerance=(1.85, 2.15), sigma_threshold=5):
    result = np.full(magnitudes_array.shape, False)
    upper_lim = -2.5 * np.log10(tolerance[0])
    lower_lim = -2.5 * np.log10(tolerance[1])
    kde_weights = np.power(errors_array, -2)
    kde_result = kde_pdf(magnitudes_array, kde_weights, kde_bandwidth)
    pdf, x = kde_result["pdf"], kde_result["x"]
    minima = argrelextrema(pdf, np.less)[0]
    n_minima = len(minima)

    if n_minima == 1:
        excursions = magnitudes_array[magnitudes_array < x[minima[0]]]
        n_excursions = len(excursions)

        if n_excursions == 1:
            excursion_errors = errors_array[magnitudes_array < x[minima[0]]]
            cluster_samples = magnitudes_array[magnitudes_array > x[minima[0]]]
            cluster_weights = kde_weights[magnitudes_array > x[minima[0]]]

            w_mean = weighted_mean(cluster_samples, cluster_weights)
            w_sigma = weighted_std(cluster_samples, cluster_weights)

            delta_mag = excursions[0] - w_mean
            delta_mag_sigma = excursion_errors[0] + w_sigma

            within_tolerance = lower_lim <= delta_mag <= upper_lim
            significant_delta = (np.abs(delta_mag / delta_mag_sigma)) > sigma_threshold

            if (within_tolerance) and (significant_delta):
                result = magnitudes_array == excursions[0]

    return result

def detect_stable(magnitudes_array, errors_array, kde_bandwidth=0.1254, n_detections=5):
    kde_weights = np.power(errors_array, -2)
    kde_result = kde_pdf(magnitudes_array, kde_weights, kde_bandwidth)
    n_maxima = count_n_maxima(kde_result)

    # Take n_detections out of this function and move it to the notebook
    if n_maxima == 1 and len(magnitudes_array) > n_detections:
        result = np.full(magnitudes_array.shape, True)
    else:
        result = np.full(magnitudes_array.shape, False)
        
    return result