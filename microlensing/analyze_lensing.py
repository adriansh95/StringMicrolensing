"""
This module provides functions for analyzing candidate lensing events.
"""
from collections import Counter
import numpy as np
import pandas as pd
from .helpers import get_bounding_idxs

def make_lensing_dataframe(
        df,
        time_column="mjd",
        exp_time_column="exptime",
        label_column="cluster_label"):
    """This function assumes the dataframe has been filtered using lens_filter,
    and is intended for use only on lightcurves
    with bright sequences (ie, no lightcurves which show only baseline).
    It calculates the earliest and latest start and stop times for a 
    string microlensing event."""
    column_list = [
        "objectid",
        time_column,
        exp_time_column,
        label_column,
        "filter"
    ]
    df = df.sort_values(by=time_column)
    df_grouped = df.groupby(by=["objectid"], sort=False)
    lens_df = df_grouped[column_list].apply(_lens_apply)

    if lens_df.shape[0] == 0:
        result = pd.DataFrame(
            columns=[
                "t_start_max",
                "t_end_max",
                "t_start_min",
                "t_end_min",
                "n_u",
                "n_g",
                "n_r",
                "n_i",
                "n_z",
                "n_Y",
                "n_samples"
            ],
            index=pd.MultiIndex.from_arrays(
                [[], []],
                names=["objectid", "event_number"]
            )
        )
    else:
        result = lens_df

    return result

def _lens_apply(df):
    s_per_day = 86400
    cl_array = df.iloc[:, 3].to_numpy()
    bounding_idxs = get_bounding_idxs(cl_array)
    t_start_idxs = bounding_idxs[:, 0]
    t_end_idxs = bounding_idxs[:, 1]
    t_start_min = df.iloc[t_start_idxs + 1, 1].to_numpy()
    t_end_min = (
        df.iloc[t_end_idxs - 1, 1] +
        df.iloc[t_end_idxs - 1, 2] / s_per_day
    ).to_numpy()

    if t_start_idxs[0] == -1:
        t_start_max = np.concatenate(
            [
                [-np.inf],
                (
                    df.iloc[t_start_idxs[1:], 1] +
                    (df.iloc[t_start_idxs[1:], 2] / s_per_day)
                ).to_numpy()
            ]
        )
    else:
        t_start_max = (
            df.iloc[t_start_idxs, 1] +
            (df.iloc[t_start_idxs, 2] / s_per_day)
        ).to_numpy()

    if t_end_idxs[-1] == len(cl_array):
        t_end_max = np.concatenate([df.iloc[t_end_idxs[:-1], 1].values, [np.inf]])
    else:
        t_end_max = df.iloc[t_end_idxs, 1].to_numpy()

    filter_counters = [
        Counter(df.iloc[idx[0]+1: idx[1], 4]) for idx in bounding_idxs
    ]
    count_data = {
        f"n_{f}": [c.get(f, 0) for c in filter_counters]
        for f in ['u', 'g', 'r', 'i', 'z', 'Y']
    }
    count_data["n_samples"] = [(idx[1] - idx[0]) - 1 for idx in bounding_idxs]
    t_data = {
        "t_start_max": t_start_max,
        "t_end_max": t_end_max, 
        "t_start_min": t_start_min, 
        "t_end_min": t_end_min
    }
    data = t_data | count_data
    result = pd.DataFrame(data=data)
    result.index.names = ["event_number"]
    return result

def t_of_tau(taus, ts):
    """Computes the amount of time in days during which events of duration
    taus have where they could begin before between ts[0] and ts[1]
    and end between ts[2] ts[3]. Normalizing this curve by its integral
    gives a posterior distribution for the duration of the event"""
    t0, t1, t2, t3 = ts
    t_max = np.min([t1 - t0, t3 - t2])
    tau_min = t2 - t1
    tau_max = t3 - t0
    tau_med = (tau_max + tau_min) / 2
    x = taus - tau_min
    x_med = tau_med - tau_min
    x_max = tau_max - tau_min
    y = np.piecewise(x, [x < x_med, x >= x_med], [lambda xx: xx, lambda xx: x_max - xx])
    result = np.clip(y, a_min=0, a_max=t_max)
    return result

def integrated_event_duration_posterior(taus, ts):
    """integrates the posterior for an event with start/stop times bounded by ts
    in bins given by taus"""
    result = np.zeros(taus.shape)

    if ~(np.isfinite(ts).all()):
        result[-1] = 1
    else:
        t0, t1, t2, t3 = ts
        t_max = np.min([t1 - t0, t3 - t2])
        tau_min = t2 - t1
        tau_max = t3 - t0
        tau_vertices = np.array([tau_min, tau_min + t_max, tau_max - t_max, tau_max])
        x = np.concatenate((taus, tau_vertices))
        mask = np.concatenate((np.full(taus.shape, True), np.full(tau_vertices.shape, False)))
        indices = np.argsort(x)
        x = x[indices]
        mask = mask[indices]
        vertex_idxs = np.nonzero(~mask)[0]
        y = t_of_tau(x, ts)
        y_av = (y[1:] + y[:-1]) / 2
        dx = np.diff(x)
        integral = y_av * dx
        integral[np.clip(vertex_idxs - 1, a_min=0, a_max=None)] += integral[vertex_idxs]
        result[:-1] = integral[mask[:-1]]
        result /= result.sum()

    return result

def count_events_per_source(df):
    """This function works on the dataframe resulting from make_lensing_dataframe.
    It groups by objectid (level 0), and counts the number of rows within each group
    by selecting the 'filters' column and calling the 'count' aggregator supplied by
    pandas."""
    result = df.groupby(level=0).filters.agg("count")
    result.name = "n_events"
    return result

def count_filter_seq(df):
    """This function works on the dataframe resulting from make_lensing_dataframe.
    It takes unique filter keys for the bright sequence, sorts them so duplicate sequences
    like 'gri' and 'gir' are counted correctly, and counts the number of each sequence."""
    filters_in_event = df["filters"].apply(lambda x: "".join(sorted(set(x))))
    result = filters_in_event.value_counts()
    return result
