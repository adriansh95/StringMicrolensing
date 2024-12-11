import pandas as pd
import yaml
import os

from config.efficiency_config import taus
from utils.efficiency import lens_lc
from utils.kde_label import cluster_label_dataframe
from tqdm import tqdm

def main():
    sampled_windows_dir = ("/Volumes/THESIS_DATA/results/"
                           "efficiency/sampled_windows/")
    kde_lightcurves_dir = ("/Volumes/THESIS_DATA/results/"
                           "kde_labelled_lightcurves/")
    write_dir = ("/Volumes/THESIS_DATA/results/efficiency/lensed_lightcurves/"
                 "batch_lcs/")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_yaml = os.path.join(script_dir,  "../../config/efficiency.yaml")
    config_file = os.path.abspath(config_yaml)
    bw_types = ["fixed", "variable"]
    bws = [0.13, "variable"]
    read_cols = ["objectid", "filter", "mag_auto", "magerr_auto",  
                 "mjd", "exptime", "mjd_mid"]

    with open(config_file) as f:
        config = yaml.safe_load(f)

    n_batches = config["data_retrieval"]["num_batches"]
    n_events = config["synthetic_events"]["n_events"]
    n_taus = len(taus)
    versions = config["scanner"].keys()

    for i_batch in tqdm(range(16, 18)):
        lc_file = (f"{kde_lightcurves_dir}kde_labelled_lightcurves"
                   f"_batch{i_batch}.parquet")
        batch_lcs = pd.read_parquet(lc_file,
                                    columns=(read_cols)
                                    )

        for v in ["v1"]:

            for i_tau in tqdm(range(n_taus)):
                tau = taus[i_tau]

                for bw_type, bw in zip(bw_types, bws):
                    pq_file = (f"{sampled_windows_dir}{bw_type}_bw_sampled"
                               f"_windows_tau{i_tau}_{v}.parquet")

                    try:
                        windows_df = pd.read_parquet(pq_file, 
                                                     columns=["t_start"]
                                                     ).iloc[:n_events]
                    except FileNotFoundError:
                        continue

                    windows_df.reset_index(inplace=True)
                    sampled_lcs = windows_df.merge(batch_lcs, on="objectid")
                    sampled_lcs.sort_values(by=["objectid", "mjd_mid"],
                                            inplace=True)
                    g = sampled_lcs.groupby(by=["objectid", "number"],
                                            sort=False)
                    lensed_lcs = g.apply(lens_lc, tau, include_groups=False)
                    lensed_lcs.reset_index(level=[0, 1], inplace=True)
                    lensed_lcs.sort_values(by=["objectid", "mjd_mid"],
                                           inplace=True)
                    groups = ["objectid", "number", "filter"]
                    lensed_lcs = cluster_label_dataframe(lensed_lcs,
                                                         groups=groups,
                                                         bandwidth=bw)
                    fname = (f"{write_dir}lensed_lcs_batch{i_batch}_tau{i_tau}"
                             f"_{v}_{bw_type}_bw.parquet")
                    lensed_lcs.to_parquet(fname)

if __name__ == "__main__":
    main()
