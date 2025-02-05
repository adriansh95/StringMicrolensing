import yaml
import pandas as pd
from tqdm import tqdm
from config.efficiency_config import taus
from pyarrow.lib import ArrowInvalid

config_file = "/Users/adrianshestakov/Work/stringScratch/config/efficiency.yaml"
good_windows_dir = "/Volumes/THESIS_DATA/results/good_windows/"

with open(config_file) as f:
    config = yaml.safe_load(f)

versions = list(config["scanner"].keys())

for v in versions[2:]:
    for i_tau in tqdm(range(taus.shape[0])):
        dfs = []

        for i_batch in range(67):
            try:
                df = pd.read_parquet(
                    f"{good_windows_dir}good_windows_batch{i_batch}_{v}.parquet",
                    columns=[f"tau_{i_tau}"]
                ).dropna()
            except ArrowInvalid:
                continue
            dfs.append(df)

        if len(dfs) > 0:
            tau_df = pd.concat(dfs)
            tau_df.to_parquet(f"{good_windows_dir}good_windows_tau{i_tau}_{v}.parquet")
        else:
            continue
