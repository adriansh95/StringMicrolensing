import pandas as pd
import numpy as np
import yaml

from tqdm import tqdm
from utils.filtering import lightcurve_classifier

def weighted_std(vals, weights):
    result = np.sqrt(np.cov(vals, aweights=weights).item())
    return result

def lc_class_dataframe(lc, config_dict):
    versions = list(config_dict.keys())
    fixed_bw_lc_class = [lightcurve_classifier(lc,
                                               **config_dict[v],
                                               label_column="bandwidth_0.13")
                         for v in versions]
    var_bw_lc_class = [lightcurve_classifier(lc,
                                             **config_dict[v],
                                             label_column="bandwidth_variable")
                       for v in versions]
    fixed_bw_idx = [f"fixed_bw_{v}_lc_class" for v in versions]
    variable_bw_idx = [f"variable_bw_{v}_lc_class" for v in versions]
    idx = fixed_bw_idx + variable_bw_idx
    data = fixed_bw_lc_class + var_bw_lc_class
    result = pd.DataFrame(data=data, index=idx)
    return result

def write_batch_summary_table(read_dir, write_dir, i_batch, config_dict):
    p = f"{read_dir}kde_labelled_lightcurves_batch{i_batch}.parquet"
    df = pd.read_parquet(p, columns=["objectid",
                                     "mag_auto",
                                     "magerr_auto",
                                     "mjd",
                                     "mjd_mid",
                                     "exptime",
                                     "filter",
                                     "bandwidth_variable",
                                     "bandwidth_0.13"]
                         ).sort_values(by=["objectid", "mjd"])
    g1 = df.groupby(by="objectid", sort=False)
    g2 = df.groupby(by=["objectid", "filter"], sort=False)
    lc_class = g1.apply(lc_class_dataframe,
                        config_dict,
                        include_groups=False).unstack()
    lc_class.columns = lc_class.columns.get_level_values(1)
    g_sig = g2[["mag_auto", "magerr_auto"]]
    g_rms = g2["magerr_auto"]
    sig = g_sig.apply((lambda x: weighted_std(x["mag_auto"],
                                              x["magerr_auto"]**-2)))
    sig = sig.unstack()
    sig.columns = [f"std_{f}" for f in sig.columns]
    rms_err = g_rms.apply(lambda x: np.sqrt(np.average(x**2))).unstack()
    rms_err.columns = [f"rms_err_{f}" for f in rms_err.columns]
    summary_df = pd.concat([sig, rms_err, lc_class], axis=1)
    summary_df.to_parquet(f"{write_dir}summary_batch{i_batch}.parquet")

def main():
    read_dir = "/Volumes/THESIS_DATA/results/kde_labelled_lightcurves/"
    write_dir = "/Volumes/THESIS_DATA/results/summary_table/"
    config_dir = "/Users/adrianshestakov/Work/stringScratch/config/"

    with open(f"{config_dir}efficiency.yaml") as f:
        version_config = yaml.safe_load(f)["scanner"]
            
    params_config = {k: dict(samples_per_filter=v["min_per_filter"],
                             unique_filters=v["n_filters_req"])
                     for k, v in version_config.items()}
    batches = np.arange(2, 67)

    for i_batch in tqdm(batches):
        write_batch_summary_table(read_dir, write_dir, i_batch, params_config)

if __name__ == "__main__":
    main()
