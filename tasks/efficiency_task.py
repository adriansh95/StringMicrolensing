"""
This module contains the EfficiencyTask class which iterates over the
lensed lightcurves and computes some quantities useful for calculating
efficiency and quality.
"""
import os
import glob
import pandas as pd
import numpy as np
from microlensing.filtering import lightcurve_classifier
from microlensing.helpers import get_bounding_idxs
from pipeline.etl_task import ETLTask
from config.config_loader import load_config

class EfficiencyTask(ETLTask):
    """
    This class reads in lightcurves with synthetic lensing events
    from parquet files and counts how many pass detection criteria,
    how many samples are correctly labeled, and how many times an event
    is "split" by a mislabelled sample.

    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.
        config_paths (dict): Dictionary containing key, value pairs 
                             ("yaml_path", (str)) and ("python_path", (str))
                             which point to the config yaml and python files.
    Methods:
        get_extract_file_path(version, i_tau, bandwidth_type):
            Returns the file_path to process given the version, event duration
            index (i_tau), and bandwidth type..
        get_load_file_path(version, i_tau, bandwidth_type):
            Returns the file_path to load given the version, event duration
            index (i_tau), and bandwidth type..
        transform(data, version, i_tau, bandwidth_type):
            Computes the weighted std per band, rms error per band, and 
            lc_class for each source for given achromaticity version,
            event duration index (i_tau), and bandwidth type..
        run(): 
            Executes the ETL process for each file, applying extract,
            transform, and load sequentially.
        lc_class_dataframe: 
            Helper function for transform which computes the lightcurve
            class for different versions of the achromaticy requirement
            for both variable and fixed 130 mmag bandwidths.

    """

    def __init__(self, extract_dir, load_dir, config_paths):
        super().__init__(extract_dir, load_dir)
        self.config = load_config(
            yaml_path=config_paths["yaml_path"]
        )["scanner"]

    def transform(self, data, version, i_tau, bandwidth_type):
        g = data.groupby(by=["objectid", "number"], sort=False)
        lc_class_df = g.apply(
            lambda x: lightcurve_classifier(x, **self.config[version])
        )
        n_detections = sum(lc_class_df == "background")
        n_injections = g.ngroups
        n_correctly_labeled = (
            data["cluster_label"] == data["true_label"]
        ).sum()
        n_samples = data.shape[0]
        n_windows = g["cluster_label"].apply(
            lambda x: len(get_bounding_idxs(x.to_numpy()))
        )
        n_splits = (n_windows - 1).sum()
        data = {
            "n_detections": [n_detections],
            "n_injections": [n_injections],
            "n_correctly_labeled": [n_correctly_labeled],
            "n_samples": [n_samples],
            "n_splits": [n_splits]
        }
        idx = pd.MultiIndex.from_tuples(
            [(version, i_tau, bandwidth_type)],
            names=["version", "tau_index", "bandwidth_type"]
        )
        result = pd.DataFrame(data=data, index=idx)
        return result

    def run(self, **kwargs):
        """
        Run the task. It accepts the following keyword arguments:
            versions: (list, optional, default: self.config.keys()):
                A list specifying which versions of the detection
                algorithm to use.
            tau_range: (tuple, optional, default: (0, 48)): A tuple 
                specifying the range of tau indices to process. The first
                element specifies the starting index (inclusive) and the second
                specifies the last (inclusive).
        """
        versions = kwargs.get("versions", list(self.config.keys()))
        tau_range = kwargs.get("tau_range", (0, 49))
        tau_array = np.arange(tau_range[0], tau_range[1]+1)
        bandwidth_types = ["fixed", "variable"]
        kwargs["iterables"] = [versions, tau_array, bandwidth_types]
        kwargs["extract"] = {
            "columns": [
                "objectid",
                "mag_auto",
                "magerr_auto",
                "mjd",
                "mjd_mid",
                "exptime",
                "filter",
                "bandwidth_variable",
                "bandwidth_0.13"
            ]
        }
        super().run(**kwargs)

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(f"{self.load_dir}efficiency_results*tau*.parquet")
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}efficiency_results.parquet")

    def get_extract_file_path(self, version, i_tau, bandwidth_type):
        """
        Get the extract file path corresponding to the given version,
        event duration (specified indirectly by i_tau),
        and bandwidth type
        """
        result = os.path.join(
            self.extract_dir,
            f"lensed_lcs_tau{i_tau}_{version}_{bandwidth_type}_bw.parquet"
        )
        return result

    def get_load_file_path(self, version, i_tau, bandwidth_type):
        """
        Get the extract file path corresponding to the given version,
        event duration (specified indirectly by i_tau),
        and bandwidth type
        """
        result = os.path.join(
            self.load_dir,
            (
                f"efficiency_results_{version}_tau{i_tau}"
                f"_{bandwidth_type}_bw.parquet"
            )
        )
        return result
