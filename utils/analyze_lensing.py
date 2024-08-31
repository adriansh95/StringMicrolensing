"""
This module provides the make_lensing_dataframe, which is intended for use
after lightcurves have been KDE labeled and filtered. 
"""
import numpy as np
import pandas as pd
from .helpers import get_bounding_idxs

def make_lensing_dataframe(df, time_column="mjd", exp_time_column="exptime"):
    """This function assumes the dataframe has been sorted by the time_column
    and filtered using lens_filter, and is intended for use only on lightcurves
    with bright sequences (ie, no lightcurves which show only baseline).
    It calculates the earliest and latest start and stop times for a 
    string microlensing event."""
    column_list = ["objectid", time_column, exp_time_column, "cluster_label", "filter"]
    df_grouped = df[column_list].groupby(by=["objectid"], sort=False)
    result = df_grouped.apply(_lens_apply)
    return result

def _lens_apply(df):
    s_per_day = 86400
    cl_array = df["cluster_label"].values
    df_indices = df.index
    n_samples = len(cl_array)
    bounding_idxs = get_bounding_idxs(cl_array)
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

    filters = [''.join(df.loc[df_indices[idx_pair[0] + 1: idx_pair[1]], "filter"])
               for idx_pair in bounding_idxs]
    data = {"t_start_max": t_start_max,
            "t_end_max": t_end_max, 
            "t_start_min": t_start_min, 
            "t_end_min": t_end_min, 
            "filters": filters}
    result = pd.DataFrame(data=data)
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
