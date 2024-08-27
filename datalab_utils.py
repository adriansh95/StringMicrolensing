import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from numba import njit
from collections import defaultdict

FLUX_DOUBLE = -2.5 * np.log10(2)
SECONDS_PER_DAY = 86400
FILTERS = np.array(['u', 'g', 'r', 'i', 'z', 'Y', "VR"])
FILTER_ORDER = {f: i for i, f in enumerate(FILTERS)}

def _filter_map(char):
    result = FILTER_ORDER[char]
    return result

filter_map = np.vectorize(_filter_map, otypes=[np.int32])

def _filter_mask_to_str(mask):
    result = "".join(FILTERS[mask])
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
    l = ["objectid", "filter", mag_column, magerr_column]
    g = df[l].groupby(by=["objectid", "filter"], sort=False, group_keys=False)
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
    lensed_idxs = _get_bounding_idxs(cl)
    n_windows = len(lensed_idxs)

    if n_windows > 0:
        df_index = df.index
        
        if check_achromatic:
            achromatic = [_check_achromaticity(df, df_index[idx_pair[0]+1: idx_pair[1]]) for idx_pair in lensed_idxs]
        else:
            achromatic = np.full(n_windows, True)
    
        if check_factor_of_two:
            g = df.groupby(by="filter", sort=False)
            factor_of_two = [_check_factor(df, g, df_index[idx_pair[0]+1: idx_pair[1]], **kwargs) for idx_pair in lensed_idxs]
        else:
            factor_of_two = np.full(n_windows, True)
    
        enough_samples = [idx_pair[1] - idx_pair[0] > min_n_samples for idx_pair in lensed_idxs]
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
    sig0 = weighted_std_err(weights[mask_idxs])
    mu1 = np.average(samples[mask_baseline], weights=weights[mask_baseline])
    sig1 = weighted_std_err(weights[mask_baseline])
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
    s_per_day = SECONDS_PER_DAY
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
    baseline = np.average(samples_baseline, weights=weights_baseline)
    baseline_err = np.sqrt(1 / weights_baseline.sum())
    result = df.assign(delta_mag=df[mag_column] - baseline,
                       delta_mag_err=df[magerr_column] + baseline_err)
    return result

def unimodal_filter(df):
    result = (df["cluster_label"].values.astype(bool)).all()
    return result

@njit
def _compute_t_start(exposure_ends, start_idx, end_idxs, taus):
    # This gives the earlist time after mjds[start_idx] + exp_times
    condition = exposure_ends[start_idx] > (exposure_ends[end_idxs] - taus)
    result = np.where(condition, exposure_ends[start_idx], exposure_ends[end_idxs] - taus)
    return result

def _delta_t_ends(mjds, t_starts, t_end_idxs, taus):
    result = mjds.take(t_end_idxs + 1, mode="clip") - (t_starts + taus)
    return result

@njit
def _good_window(bright_filter_mask, baseline_filter_mask):
    enough_filters = bright_filter_mask.sum(axis=1) > 1
    # In case I choose not to use Numba here,
    #     bright_and_baseline = (~bright_filter_mask | baseline_filter_mask).all(axis=1)
    # is equivalent but quite a bit slower. 
    m = ~bright_filter_mask
    bright_and_baseline = np.array([(m[i] | baseline_filter_mask[i]).all() 
                                    for i in range(len(m))])
    result = enough_filters & bright_and_baseline
    return result

@njit
def _keep_scanning(t_start_idxs, t_end_idxs, n_samples):
    # This function returns a boolean array, true when the last
    # bright sample in the window is not the last sample in the LC
    # AND when the first bright sample is not the penultimate sample
    # in the LC (cannot place an upper bound on the window)
    result = (t_start_idxs < n_samples - 3) & (t_end_idxs < n_samples - 1)
    return result

def effective_monitoring_time(lc_df, taus):
    result = defaultdict(lambda: np.zeros(len(taus)))

    for group_name, lc in lc_df.groupby(by="objectid", sort=False):
        mjds = lc["mjd"].values
        filters = lc["filter"].values
        exp_times = lc["exptime"].values / SECONDS_PER_DAY
        exposure_ends = mjds + exp_times
        _effective_monitoring_time(mjds, filters, exposure_ends, taus, result, group_name)

    return result

def _effective_monitoring_time(mjds, filters, exposure_ends, taus, results_dict, group_name):
    n_filters = len(FILTER_ORDER.keys())
    n_taus = len(taus)
    n_samples = len(mjds)
    tau_idx = np.arange(n_taus)
    mjd_idxs = np.arange(n_samples)
    n_filters_all = np.zeros(n_filters, dtype=np.int32)
    np.add.at(n_filters_all, filter_map(filters), 1)

    t_start_idxs = np.zeros(n_taus, dtype=np.int32)
    initial_t_start = exposure_ends[t_start_idxs]
    t_end_idxs = np.array([mjd_idxs[mjds < initial_t_start[i] + taus[i]][-1] for i in tau_idx]) #Vectorize?
    t_starts = _compute_t_start(exposure_ends, t_start_idxs, t_end_idxs, taus)

    n_filters_bright = np.zeros((n_taus, n_filters), dtype=np.int32)

    #Can probably Vectorize this but it's not even close to being a bottleneck
    for i in tau_idx:
        np.add.at(n_filters_bright[i], filter_map(filters[1: t_end_idxs[i] + 1]), 1)

    n_filters_baseline = n_filters_all - n_filters_bright

    # Start moving the windows through the lightcurve
    keep_scanning = _keep_scanning(t_start_idxs, t_end_idxs, n_samples)

    while keep_scanning.any():
        # Compute time between beginning of t_start and first bright sample
        delta_t_starts = mjds[t_start_idxs + 1] - t_starts
        # Find the difference between the end of the bright window and the
        # the first baseline sample after the bright window
        # This is negative if the window ends after the last
        # Sample in the lightcurve
        delta_t_ends = _delta_t_ends(mjds, t_starts, t_end_idxs, taus)
        # There's some amount of time the window can safely move to the right 
        # along The lightcurve before the start or end hits another sample
        # This keeps track of whether its the start of the window that hits
        # a sample first, or the end.
        smaller_delta_t_start = delta_t_starts < delta_t_ends
        # This has shape (tau,). We will add this "safe" time to results_dict
        # if the window has not extended past the end of the lightcurve 
        # AND the window has enough bright samples inside 
        # and baseline samples outside.
        t_add = np.maximum(np.where(smaller_delta_t_start, 
                                    delta_t_starts, delta_t_ends), 0)

        # Now we check that there's enough baseline and bright samples
        # in the window
        bright_filter_mask = n_filters_bright > 0
        baseline_filter_mask = n_filters_baseline > 0
        good_window = _good_window(bright_filter_mask, baseline_filter_mask)

        m = keep_scanning & good_window
        # assert (t_add[6] == 0 or ~m[6]), (f"""
        # {t_add[6]}, {group_name}, {t_start_idxs[6]}, {t_end_idxs[6]}, 
        # {filters[9:11]}, {m[6]}
        # """)

        for i, t in zip(tau_idx[m], t_add[m]):
            k = _filter_mask_to_str(bright_filter_mask[i])
            results_dict[k][i] += t

        # Begin the process of shifting the window past the incoming (outgoing)
        # sample by changing the bright and baseline filters counters.
        # Get the index of the next filter for each tau. 
        # This is t_start_idxs + 1
        # if the next sample is outgoing, t_end_idxs + 1 otherwise.
        next_filter_idx = np.where(smaller_delta_t_start, t_start_idxs + 1, t_end_idxs + 1)
        # Safely get the "next" filter using these idxs. If the bright window
        # Has extended beyond the end of the lightcurve, this will just get
        # The last filter in the lightcurve and will be masked out.
        next_filter = filters.take(next_filter_idx, mode="clip")
        next_filter_number = filter_map(next_filter)[keep_scanning]
        # If the next filter is outgoing (smaller_delta_t_start) subtract from
        # n_filters_bright and add to n_filters_baseline
        next_filter_change = np.where(smaller_delta_t_start, -1, 1)[keep_scanning]
        tau_idx_masked = tau_idx[keep_scanning]
        n_filters_bright[tau_idx_masked, next_filter_number] += next_filter_change
        n_filters_baseline[tau_idx_masked, next_filter_number] -= next_filter_change

        # Keeping track of which samples are inside the lightcurve. Add 1 to t_start_idx
        # if the next sample is outgoing. Add 1 to t_end_idx
        # if the next sample is incoming.
        start_idx_shift = np.where(smaller_delta_t_start, 1, 0)[keep_scanning]
        t_start_idxs[keep_scanning] += start_idx_shift
        t_end_idxs[keep_scanning] -= start_idx_shift - 1

        # Recompute t_starts for the next iteration, 
        t_starts = _compute_t_start(exposure_ends, t_start_idxs, t_end_idxs, taus)
        # Determine which windows are still within the lightcurve
        keep_scanning = _keep_scanning(t_start_idxs, t_end_idxs, n_samples)

