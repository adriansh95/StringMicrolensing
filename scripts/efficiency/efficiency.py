import pandas as pd
import yaml
import os
import glob

from config.efficiency_config import taus
from utils.efficiency import make_efficiency_df
from tqdm import tqdm

def write_efficiency_dfs(read_dir, write_dir, config, n_taus):
    bw_types = ["fixed", "variable"]

    for v, scanner_params in config["scanner"].items():
        if v == "v0":
            continue
        params = dict(samples_per_filter=scanner_params["min_per_filter"],
                      unique_filters =scanner_params["n_filters_req"])

        for i_tau in tqdm(range(n_taus)):

            for bw in bw_types:
                lc_file = f"{read_dir}lensed_lcs_tau{i_tau}_{v}_{bw}_bw.parquet"

                try:
                    lc_df = pd.read_parquet(lc_file)
                except FileNotFoundError:
                    continue

                idx = pd.MultiIndex.from_tuples([(v, i_tau, bw)],
                                                names=["version", "tau_index",
                                                       "bandwidth_type"])
                e_df = make_efficiency_df(lc_df, idx, **params)
                fname = f"{write_dir}{v}_tau{i_tau}_{bw}_bw.parquet"
                e_df.to_parquet(fname)

def concat_efficiency_dfs(df_dir):
    efficiency_df_files = glob.glob(f"{df_dir}*tau*.parquet")
    efficiency_dfs = [pd.read_parquet(f) for f in efficiency_df_files]
    concatted_df = pd.concat(efficiency_dfs, axis=0)
    concatted_df.sort_index(inplace=True)
    concatted_df.to_parquet(f"{df_dir}efficiency_results.parquet")

def main(read_dir, write_dir, config, n_taus):
    #write_efficiency_dfs(read_dir, write_dir, config, n_taus)
    concat_efficiency_dfs(write_dir)

if __name__ == "__main__":
    read_dir = ("/Volumes/THESIS_DATA/results/efficiency/lensed_lightcurves/"
                "concatted_lcs/")
    write_dir = "/Volumes/THESIS_DATA/results/efficiency/output_data/"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_relpath = os.path.join(script_dir,  "../../config/efficiency.yaml")
    config_file = os.path.abspath(config_relpath)
    n_taus = len(taus)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    main(read_dir, write_dir, config, n_taus)

