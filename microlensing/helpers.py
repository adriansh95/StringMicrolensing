"""
This module provides helper functions which are shared amongst
more than one module in the same directory
"""
import numpy as np

FILTER_ORDER = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5, "VR": 6}

def weighted_std(vals, weights):
    var = np.cov(vals, aweights=weights).item()
    result = np.sqrt(var)
    return result

def filter_map(char):
    result = FILTER_ORDER[char]
    return result

def get_bounding_idxs(cluster_label_array):
    """This function finds time-contiguous sequences of measurements within
    a lightcurve which were labelled bright by the gaussian KDE method
    within the kde_labeling module and returns the bounding indexes 
    (that is, the index of the sample preceding the bright sequence
    and the index of the sample following the bright sequence)."""
    n_total = len(cluster_label_array)
    idxs = np.arange(n_total)
    t_start = [i for i in idxs[:-1]
               if cluster_label_array[i] != 0
               and cluster_label_array[i+1] == 0]
    t_end = [i+1 for i in idxs[:-1]
             if cluster_label_array[i] == 0
             and cluster_label_array[i+1] != 0]

    if cluster_label_array[0] == 0:
        t_start = np.concatenate(([-1], t_start))
    if cluster_label_array[-1] == 0:
        t_end = np.concatenate((t_end, [n_total]))

    result = np.column_stack([t_start, t_end]).astype(int)
    return result

def subtract_baseline(df_grouped, mag_column="mag_auto", magerr_column="magerr_auto"):
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

def write_query(i_batch, batch_size, db):
    sub_query = f"""
    SELECT id 
        FROM {db}
        WHERE row_number BETWEEN {i_batch * batch_size} AND {(i_batch + 1) * batch_size - 1}
    """
    result = f"""
    SELECT m.objectid, m.filter, m.mag_auto, m.magerr_auto, m.mjd, m.exposure, e.exptime
        FROM nsc_dr2.meas AS m
        INNER JOIN nsc_dr2.exposure AS e
        ON e.exposure = m.exposure
        WHERE m.objectid IN ({sub_query})
    """
    return result
