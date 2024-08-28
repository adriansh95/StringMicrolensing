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
    with bright sequences (ie, no lightcurves which show only baseline)."""
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
