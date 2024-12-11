import os
import yaml
import dask.dataframe as dd

from utils.efficiency import t_start_df
from tqdm import tqdm

def filter_rows(df, ids):
    result = df.loc[df.index.get_level_values(0).isin(ids)]
    return result

def main():
    good_windows_dir = "/Volumes/THESIS_DATA/results/good_windows/"
    summary_table_dir = "/Volumes/THESIS_DATA/results/summary_table/"
    write_dir = "/Volumes/THESIS_DATA/results/efficiency/sampled_windows/"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_yaml = os.path.join(script_dir,  "../../config/efficiency.yaml")
    config_file = os.path.abspath(config_yaml)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    n_samples = config["synthetic_events"]["n_events"]

    for v in ["v1"]:
        good_window_files = [os.path.join(good_windows_dir, 
                                          f"good_windows_batch{i}_{v}.parquet")
                             for i in range(16, 18)]
        ddf = dd.read_parquet(good_window_files)
        summ_parquets = [f"{summary_table_dir}summary_batch{i}.parquet"
                         for i in range(67)]
        lc_class_ddf = dd.read_parquet(summ_parquets,
                                       columns=[f"fixed_bw_{v}_lc_class",
                                                f"variable_bw_{v}_lc_class"])
        columns = ddf.columns
        fixed_lc_class = lc_class_ddf[f"fixed_bw_{v}_lc_class"]
        fixed_stable = fixed_lc_class == "unimodal"
        fixed_stable_ids = fixed_lc_class.loc[fixed_stable].compute()
        fixed_stable_ids = fixed_stable_ids.index.to_numpy()
        var_lc_class = lc_class_ddf[f"variable_bw_{v}_lc_class"]
        var_stable = var_lc_class == "unimodal"
        var_stable_ids = var_lc_class.loc[var_stable].compute()
        var_stable_ids = var_stable_ids.index.to_numpy()


        for col in tqdm(columns):
            try:
                filtered_df = ddf[col].dropna().compute()
            except KeyError:
                continue

            for ids, bw_type in zip([fixed_stable_ids, var_stable_ids],
                                    ["fixed_bw", "variable_bw"]):
                m = filtered_df.index.get_level_values(0).isin(ids)
                stable_df = filtered_df.loc[m]
                sampled_df = t_start_df(stable_df, n_samples)
                c = ''.join(col.split('_'))
                fname = f"{bw_type}_sampled_windows_{c}_{v}.parquet"
                sampled_df.to_parquet(os.path.join(write_dir, fname))

if __name__ == "__main__":
    main()
