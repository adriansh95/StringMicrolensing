"""
This module contains the SummaryTableTask class which iterates over the
batched lightcurves and computes some useful statistics
(weighted standard deviation, RMS error on the magnitudes, and lightcurve
class).
"""
import os
import glob
import pandas as pd
import numpy as np
from microlensing.filtering import lightcurve_classifier
from microlensing.helpers import weighted_std
from pipeline.etl_task import ETLTask
from config.config_loader import load_config

class SummaryTableTask(ETLTask):
    """
    This class reads in kde labelled lightcurves from parquet files
    and computes some the weighted standard deviation of the measurements
    in each band, labels the lightcurve as 'unimodal', 'background', 
    'unstable', or 'NA', and computes the rms error of the magnitudes per band.
    This is done per source, and the resulting dataframe is saved to the
    load_directory for each batch.

    Attributes:

        extract_dir (str): 
            Directory containing the input data files to be  processed.

        load_dir (str): 
            Directory where the transformed data files will be  written.

        config_paths (dict): 
            Dictionary containing key, value pairs  ("yaml_path", (str))
            and ("python_path", (str)) which point to the config yaml 
            and python files. Note: python_path not used.
    Methods:

        get_extract_file_path(i_batch):
            Returns the file_path to process given the batch number.

        get_load_file_path(i_batch):
            Returns the file_path to load given the batch number.

        extract(data_file_path):
            Reads data from a given input file and returns it as a DataFrame.

        transform(data):
            Computes the weighted std per band, rms error per band, and 
            lc_class for each source.

        load(data, data_file_path): 
            Writes the transformed DataFrame to the specified output directory.

        run(): 
            Executes the ETL process for each file, applying extract,
            transform, and load sequentially.

        lc_class_dataframe(data): 
            Helper function for transform which computes the lightcurve
            class for different versions of the achromaticy requirement
            for both variable and fixed 130 mmag bandwidths.

    """

    def __init__(self, extract_dir, load_dir, config_paths):
        super().__init__(extract_dir, load_dir)
        self.config = load_config(
            yaml_path=config_paths["yaml_path"]
        )["scanner"]

    def lc_class_dataframe(self, data):
        """
        Loops over different versions of achromaticity requirements
        and labels the lightcurves as "unimodal", "background",
        "unstable", or "NA".
        """
        versions = list(self.config.keys())
        fixed_bw_lc_class = [
            lightcurve_classifier(
                data,
                label_column="bandwidth_0.13",
                **self.config[v]
            )
            for v in versions
        ]
        var_bw_lc_class = [
            lightcurve_classifier(
                data,
                label_column="bandwidth_variable",
                **self.config[v]
            )
            for v in versions
        ]
        fixed_bw_idx = [f"fixed_bw_{v}_lc_class" for v in versions]
        variable_bw_idx = [f"variable_bw_{v}_lc_class" for v in versions]
        idx = fixed_bw_idx + variable_bw_idx
        result_data = fixed_bw_lc_class + var_bw_lc_class
        result = pd.DataFrame(data=result_data, index=idx)
        return result

    def transform(self, data, *args):
        """
        Transform the data. *args added in signature for
        compatibility with ETLTask
        """
        g1 = data.groupby(by="objectid", sort=False)
        lc_class = g1.apply(
            self.lc_class_dataframe,
            include_groups=False
        ).unstack()
        lc_class.columns = lc_class.columns.get_level_values(1)

        g2 = data.groupby(by=["objectid", "filter"], sort=False)
        sig = g2[["mag_auto", "magerr_auto"]].apply(
            lambda x: weighted_std(x["mag_auto"], x["magerr_auto"]**-2)
        )
        sig = sig.unstack()
        sig.columns = [f"std_{f}" for f in sig.columns]

        rms_err = g2["magerr_auto"].apply(
            lambda x: np.sqrt(np.average(x**2))
        )
        rms_err = rms_err.unstack()
        rms_err.columns = [f"rms_err_{f}" for f in rms_err.columns]
        result = pd.concat(
            [sig, rms_err, lc_class],
            axis=1
        )
        return result

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(
            f"{self.load_dir}summary_batch*.parquet"
        )
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}summary_table.parquet")

    def run(self, **kwargs):
        """
        Run the task. It accepts the following keyword arguments:
            batch_range: (tuple, optional): A tuple specifying the range 
                of batch indices to process. Defaults to (0, 66). The first
                element specifies the starting index (inclusive) and the second
                specifies the last (inclusive).
        """
        batch_range = kwargs.get("batch_range", (0, 66))
        first_batch = batch_range[0]
        last_batch = batch_range[1]
        batch_array = np.arange(first_batch, last_batch+1)
        kwargs["iterables"] = [batch_array]
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

    def get_extract_file_path(self, *args):
        """Get the extract file path corresponding to i_batch"""
        i_batch, = args
        result = os.path.join(
            self.extract_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
        )
        return result

    def get_load_file_path(self, *args):
        """Get the load file path corresponding to i_batch"""
        i_batch, = args
        result = os.path.join(
            self.load_dir,
            f"summary_batch{i_batch}.parquet"
        )
        return result
