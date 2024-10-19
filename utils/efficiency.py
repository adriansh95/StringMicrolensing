import numpy as np
import pandas as pd
import os
import time

from dl import queryClient as qc
from dl.helpers.utils import convert
from utils.kde_label import cluster_label_dataframe
from utils.helpers import get_bounding_idxs
from pyarrow.lib import ArrowInvalid

def parquet_to_mydb(read_dir, i_tau, version):
    batches = np.arange(67)
    schema = "objectid,text\nnumber,int\nstart,real\nend,real"
    qc.mydb_drop(f"windows_temp")
    qc.mydb_create(f"windows_temp", schema, drop=True)

    for batch in batches:
        df_file = os.path.join(read_dir, 
                               f"good_windows_batch{batch}_{version}.parquet")

        try:
            window_df = pd.read_parquet(df_file, columns=[f"{i_tau}"]).dropna()
        except ArrowInvalid as e:
            continue

        t_df = pd.concat([window_df.xs("start", level=2),
                          window_df.xs("end", level=2)],
                         axis=1, keys=["start", "end"])
        t_df.reset_index(inplace=True)
        t_df.columns = t_df.columns.get_level_values(0)
        qc.mydb_insert(f"windows_temp", 
                       t_df.to_csv(index=False), schema=schema)

def t_start_table(n_windows, i_tau, version, drop=False):
    id_q = "SELECT objectid FROM mydb://numbered_stable_stars_sep2"
    sq = f"""
    SELECT *
        FROM mydb://windows_temp
        WHERE objectid IN ({id_q})
        ORDER BY RANDOM()
        LIMIT {n_windows}"""
    query = f"""
    SELECT 
        (ROW_NUMBER() OVER () - 1) AS rn,
        w.objectid, 
        w.number, 
        w.start + (RANDOM() * (w.end - w.start)) AS t_start
    FROM ({sq}) AS w"""
    qc.query(sql=query, out=f"mydb://t_start_{i_tau}_{version}", drop=drop)
    qc.mydb_index(f"t_start_{i_tau}_{version}", "objectid")
    qc.mydb_index(f"t_start_{i_tau}_{version}", "rn")

def make_lightcurve_dataframe(i_tau, i_batch, batch_size, version):
    sq = f"""
    SELECT *
        FROM mydb://t_start_{i_tau}_{version}
        WHERE rn BETWEEN {i_batch * batch_size} 
        AND {(i_batch + 1) * batch_size - 1}
    """
    qc.query(sql=sq, out="mydb://temp", drop=True)
    qc.mydb_index("temp", "objectid")
    q = f"""
    SELECT 
        t.objectid, t.number, t.t_start, 
        m.filter, m.mag_auto, m.magerr_auto, m.mjd,
        m.exptime, (m.mjd + (m.exptime / (86400 * 2))) AS mjd_mid
        FROM mydb://stable_lcs AS m
        INNER JOIN mydb://temp AS t
        ON t.objectid = m.objectid
    """
    job_id = qc.query(sql=q, async_=True, timeout=3600)

    while qc.status(job_id) == "EXECUTING":
        time.sleep(10)

    if qc.status(job_id) == "COMPLETED":
        result = convert(qc.results(job_id))
        result.sort_values(by=["objectid", "mjd"], inplace=True)
    elif qc.status(job_id) == "ERROR":
        print(f"Error batch {i_batch}: {qc.error(job_id)}")
        result = None
    else:
        print(f"Error batch {i_batch}: Something unexpected occurred")
        result = None

    qc.mydb_drop("temp")
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

def count_windows(df):
    cl = df["cluster_label"].to_numpy()
    result = len(get_bounding_idxs(cl))
    return result