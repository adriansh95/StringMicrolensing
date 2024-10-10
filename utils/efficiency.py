import numpy as np
import pandas as pd

from dl import queryClient as qc
from dl.helpers.utils import convert
from utils.kde_label import cluster_label_dataframe
from utils.filtering import lens_filter

def make_lens_df(unimodal_ids, tau_window):
    mask = tau_window.index.isin(unimodal_ids["objectid"], level=0)
    # print(unimodal_ids, tau_window.index.get_level_values(0), mask.any())
    # mask = np.full(len(tau_window), True) #DELETE
    rng = np.random.default_rng()
    masked_windows_df = tau_window.loc[mask]
    weight_df = (masked_windows_df.xs("end", level=2)
                 - masked_windows_df.xs("start", level=2))
    oids = weight_df.index.get_level_values(0)
    win_nums = weight_df.index.get_level_values(1)
    weights = weight_df.values / weight_df.sum()
    all_choices = list(zip(oids, win_nums))
    choices = rng.choice(all_choices, size=1000, p=weights)
    choices = [(c[0], int(c[1])) for c in choices]
    start = masked_windows_df.xs("start", level=2)
    end = masked_windows_df.xs("end", level=2)
    low = start.loc[choices]
    high = end.loc[choices]
    t_start = rng.uniform(low=low.values, high=high.values)
    lens_data = np.column_stack(([c0 for c0, c1 in choices],
                                  t_start))
    result = pd.DataFrame(data=lens_data, columns=["objectid", "t_start"])
    result["copy"] = result.groupby(by="objectid").cumcount()
    return result

def make_temp_table(lens_df):
    schema = "objectid,text\ncopy,int\nt_start,float8"
    l = ["objectid", "copy", "t_start"]
    qc.mydb_drop("temp")
    qc.mydb_create("temp", schema, drop=True)
    qc.mydb_insert("temp", lens_df[l].to_csv(index=False), schema=schema)

def make_lightcurve_dataframe():
    sq = "SELECT * FROM mydb://temp"
    q = """SELECT t.objectid, t.copy, t.t_start,
               m.filter, m.mag_auto, m.magerr_auto, m.mjd, e.exptime
           FROM nsc_dr2.meas AS m
           INNER JOIN mydb://temp AS t
           ON t.objectid = m.objectid
           INNER JOIN nsc_dr2.exposure AS e
           ON e.exposure = m.exposure
    """
    result = qc.query(sql=q, fmt="pandas")
    result["mjd_mid"] = result["mjd"] + (result["exptime"] / (86400 * 2))
    result.sort_values(by=["objectid", "mjd"], inplace=True)
    return result

def lens_lc(lc, tau):
    t_start = lc["t_start"].iloc[0]
    idx0, idx1 = lc["mjd_mid"].searchsorted([t_start, t_start+tau])
    vals = lc["mag_auto"].values
    vals[idx0: idx1] += -2.5 * np.log10(2)
    true_label = np.full(vals.shape, 1)
    true_label[idx0: idx1] = 0
    lc["mag_auto"] = vals
    lc["true_label"] = true_label
    return lc

def main(taus, tau_idx, unimodal_id, window_df, **kwargs):
    tau_window = window_df[tau_idx].dropna()
    lens_df = make_lens_df(unimodal_id, tau_window)
    make_temp_table(lens_df)
    lc = make_lightcurve_dataframe()
    g = lc.groupby(by=["objectid", "copy", "filter"],
                   sort=False,
                   group_keys=False)
    lc = g.filter(lambda x: len(x) > 3)
    g = lc.groupby(by=["objectid", "copy"], sort=False, group_keys=False)
    n_lensed = g.ngroups
    lensed_df = g.apply(lens_lc, (taus[tau_idx]))
    lensed_df = cluster_label_dataframe(lensed_df)
    # g = lensed_df.groupby(by=["objectid", "copy"], sort=False, group_keys=False)
    # lensed_df = g.filter(lambda group: lens_filter(group, **kwargs))
    # label_df = lensed_df[["true_label", "cluster_label"]]
    # result = label_df.set_index(["objectid", "copy", "t_start"])
    return lensed_df
