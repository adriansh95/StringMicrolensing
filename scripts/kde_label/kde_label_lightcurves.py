import pandas as pd
import numpy as np
from utils.kde_label import cluster_label_dataframe
from tqdm import tqdm

def main():
    read_dir = "/Volumes/THESIS_DATA/lightcurves/"
    write_dir = "/Volumes/THESIS_DATA/results/kde_labelled_lightcurves/"
    n_batches = 67
    bandwidths = (["variable", 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 
                   0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    ibws = [0, 13]
    col_names = [f"bandwidth_{bandwidths[ibw]}" for ibw in ibws]
    cl = "cluster_label"

    for i_batch in tqdm(range(61, n_batches)):
        parq = f"{read_dir}lightcurves_batch{i_batch}.parquet"
        df = pd.read_parquet(parq)
        g = df.groupby(by=["objectid", "filter"], group_keys=False)
        df = g.filter(lambda x: len(x) > 2)
        data = np.zeros((df.shape[0], len(ibws)))

        for i in tqdm(range(len(ibws))):
            ibw = ibws[i]
            bw = bandwidths[ibw]
            data[:, i] = cluster_label_dataframe(df, bandwidth=bw)[cl]
            #df[f"bandwidth_{bw}"] = cl["cluster_label"]

        df[col_names] = data
        fname = f"{write_dir}kde_labelled_lightcurves_batch{i_batch}.parquet"
        df.to_parquet(fname)

if __name__ == "__main__":
    main()
