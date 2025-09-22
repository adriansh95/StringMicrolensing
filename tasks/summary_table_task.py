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
    DEFAULT_ITERABLES = (0, 132)
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": [
                "objectid",
                "mag_auto",
                "magerr_auto",
                "mjd",
                "filter",
                "root_2_label",
                "2_label",
                "3_label"
            ]
        }
    }

    def lc_class_dataframe(self, data):
        """
        Loops over different versions of achromaticity requirements
        and labels the lightcurves as "unimodal", "background",
        "unstable", or "NA".
        """
        labels = ["root_2_label", "2_label", "3_label"]
        result_data = [
            lightcurve_classifier(
                data,
                label_column=l,
                min_per_filter=1,
                n_filters_req=2
            )
            for l in labels
        ]
        result = pd.DataFrame(data=result_data, index=labels)
        return result

    def transform(self, data, i_batch):
        """
        Transform the data
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
        result.index = pd.MultiIndex.from_product(
            [[i_batch], result.index],
            names=["batch_number", result.index.name]
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
        run_kwargs = {}
        batch_range = kwargs.get(
            "batch_range",
            self.DEFAULT_ITERABLES
        )
        batch_array = np.arange(batch_range[0], batch_range[1]+1, dtype=int)
        run_kwargs["iterables"] = [batch_array]
        run_kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"]
        super().run(**run_kwargs)

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
