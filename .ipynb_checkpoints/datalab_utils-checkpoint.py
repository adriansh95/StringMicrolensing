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

