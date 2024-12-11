import glob
import pandas as pd
import numpy as np
import ipdb

from utils.kde_label import _cl_apply
from tqdm import tqdm

def fix_column(group, bw):
    g = group.groupby(by="number", group_keys=False)

    if g.ngroups > 1:
        result = g.apply(_cl_apply,
                         bw,
                         "mag_auto",
                         "magerr_auto",
                         include_groups=False).iloc[:, 0].rename("cluster_label")
    else:
        result = group["cl"].rename("cluster_label")

    return result 

lc_dir = ("/Volumes/THESIS_DATA/results/efficiency/lensed_lightcurves/"
          "concatted_lcs/")
groups = ["objectid", "number", "filter"]
bw_types = ["fixed", "variable"]

for bw_type in bw_types:
    files = [f"{lc_dir}lensed_lcs_tau{i}_v2_{bw_type}_bw.parquet" for i in [28, 30]]

    if bw_type == "fixed":
        bw = lambda x: 0.13
    else:
        bw = lambda x: np.sqrt(np.mean(x**2))

    for f in tqdm(files):
        print(f)
        df = pd.read_parquet(f).rename(columns={"cluster_label": "cl"})
        has_duplicates = df.groupby(by=["objectid"])["number"].transform(lambda x: 
                                                                         x.unique().size > 1)
        df["cluster_label"] = df["cl"]
        dup_df = df.loc[has_duplicates]
        g = dup_df.groupby(by=["objectid", "filter"], group_keys=False)
        cl = g.apply(fix_column, bw, include_groups=False)
        df.loc[has_duplicates, "cluster_label"] = cl
        df.to_parquet(f)
