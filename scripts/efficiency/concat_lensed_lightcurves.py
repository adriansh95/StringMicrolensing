import pandas as pd
import yaml
import os

from tqdm import tqdm
from config.efficiency_config import taus

def main():
    lightcurve_dir = ("/Volumes/THESIS_DATA/results/efficiency/"
                      "lensed_lightcurves/")
    write_dir = f"{lightcurve_dir}concatted_lcs/"
    batch_lc_dir = f"{lightcurve_dir}batch_lcs/"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_yaml = os.path.join(script_dir,  "../../config/efficiency.yaml")
    config_file = os.path.abspath(config_yaml)
    bw_types = ["fixed_bw", "variable_bw"]

    with open(config_file) as f:
        config = yaml.safe_load(f)

    n_batches = config["data_retrieval"]["num_batches"]
    versions = config["scanner"].keys()

    for v in ["v2"]:

        for i_tau in tqdm(range(30, 31)):
            
            for bw_type in bw_types:
                lc_dfs = []
                keys = []

                for i_batch in range(n_batches):
                    f_name = (f"lensed_lcs_batch{i_batch}_tau{i_tau}_{v}"
                              f"_{bw_type}.parquet")
                    pq_file = f"{batch_lc_dir}{f_name}"

                    try:
                        lc_dfs.append(pd.read_parquet(pq_file))
                        keys.append(i_batch)
                    except FileNotFoundError:
                        continue

                if len(lc_dfs) > 0:
                    lc_df = pd.concat(lc_dfs, axis=0, keys=keys,
                                      names=["batch_number"])
                    lc_df.to_parquet((f"{write_dir}lensed_lcs_tau{i_tau}_"
                                      f"{v}_{bw_type}.parquet"))


if __name__ == "__main__":
    main()
