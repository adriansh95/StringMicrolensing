import numpy as np
import pandas as pd
import os
import time

from utils.kde_label import cluster_label_dataframe
from utils.helpers import get_bounding_idxs
from pyarrow.lib import ArrowInvalid

def t_start_df(filtered_df, n_samples):
    rng = np.random.default_rng()
    s = filtered_df.xs("start", level=2)
    e = filtered_df.xs("end", level=2)
    time_df = pd.concat([s, e], axis=1)
    time_df.columns = ["start", "end"]
    result = time_df.sample(n=n_samples)
    result["t_start"] = rng.uniform(low=result["start"], high=result["end"])
    return result

def sample_good_windows(filtered_ddf):
    s = filtered_ddf.map_partitions(lambda x: x.xs("start", level=2),
                                    meta=(None, float))
    e = filtered_ddf.map_partitions(lambda x: x.xs("end", level=2),
                                    meta=(None, float))
    time_ddf = dd.concat([s, e], axis=1)
    time_ddf.columns = ["start", "end"]
    n_rows = time_ddf.shape[0].compute()
    n_samples = 1000000

    if n_rows == 0:
        frac = 0
    else:
        frac = min(n_samples / n_rows, 1)

    sampled_ddf = time_ddf.sample(frac=frac)
    result = sampled_ddf.compute()
    return result

def lens_lc(lc, tau):
    t_start = lc["t_start"].iloc[0]
    idx0, idx1 = lc["mjd_mid"].searchsorted([t_start, t_start+tau])
    vals = lc["mag_auto"].to_numpy()
    vals[idx0: idx1] += -2.5 * np.log10(2)
    true_label = np.full(vals.shape, 1)
    true_label[idx0: idx1] = 0
    lc["mag_auto"] = vals
    lc["true_label"] = true_label
    return lc

def count_windows(df, cl_col):
    cl = df[cl_col].to_numpy()
    result = len(get_bounding_idxs(cl))
    return result

