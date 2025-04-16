import argparse
import inspect
import yaml
import numpy as np
import pandas as pd

from pathlib import Path
from rubin_sim import maf
from rubin_sim.data import get_data_dir, get_baseline
from utils.rubin_sim_utils import (
    vectorized_filter_map,
    EffectiveMonitoringTimeMetric,
)
from config.efficiency_config import taus

def main(config_yaml):
    opsim_db_fname = config_yaml.get("db_file", get_baseline())
    sim_name = Path(opsim_db_fname).stem
    output_dir = config_yaml["output_dir"]
    binned_object_df_path = config_yaml["binned_object_df_path"]
    scanner_config_file = config_yaml["scanner_config_file"]
    bounded = config_yaml["bounded"]
    object_df = pd.read_parquet(binned_object_df_path)
    ra = object_df["ra"].to_numpy()
    dec = object_df["dec"].to_numpy()
    slicer = maf.UserPointsSlicer(ra, dec)
    slicer.slice_points["count"] = object_df["count"].to_numpy()

    with open(scanner_config_file, 'r') as f:
        scanner_configs = yaml.safe_load(f)["scanner"]

    if bounded:
        b_str = "bounded"
    else:
        b_str = "unbounded"

    for v, d in scanner_configs.items():
        scanner_kwargs = {
            "n_filters_req": d["unique_filters"],
            "min_per_filter": d["samples_per_filter"],
            "bounded": bounded
        }
        metric = EffectiveMonitoringTimeMetric(taus, **scanner_kwargs)
        bundle = maf.MetricBundle(
            metric,
            slicer,
            "",
            run_name=f"{sim_name}_{v}_{b_str}"
        )
        group = maf.MetricBundleGroup(
            [bundle],
            opsim_db_fname,
            out_dir=output_dir
        )
        group.run_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate EMT using rubin_sim"
    )

    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.config, "r", encoding="utf-8") as file:
        config_yaml = yaml.safe_load(file)

    main(config_yaml)
