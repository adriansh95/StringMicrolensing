"""
This module contains the AnalyzeBackgroundsTask class which looks at 
lightcurves with lensing-like events and records 
the upper and lower bounds for the start and end times, the number of samples
per filter, and the total number of bright samples.
"""
import os
import glob
import pandas as pd
import numpy as np
from utils.filtering import lens_filter, unstable_filter
from utils.tasks.etl_task import ETLTask
from utils.analyze_lensing import make_lensing_dataframe

class AnalyzeBackgroundsTask(ETLTask):
    """
    This class reads in lightcurves from parquet files and 
    filters out those with lensing-like events. Then records the
    start and end times of a maximal and minimal length lensing event
    given the time-stamps of the lightcurve, counts the number of bright
    samples per filter, and counts the total number of bright samples
    in the lightcurve. 

    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.

    Methods:
        get_extract_file_path(i_batch):
            Returns the file_path to process given the batch number.
        get_load_file_path(i_batch):
            Returns the file_path to load given the batch number.
        transform(data, i_batch):
            Filters out lensing-like events, then records max and min
            start and stop times and counts number of samples per filter
            and total number of samples.
        concat_results():
            Combine results from batch iterations into a single dataframe.
    """

    def transform(self, data, i_batch):
        transformed_dataframes = []
        lens_filters = [
            lambda x: lens_filter(x, label_column="bandwidth_0.13"),
            lambda x: lens_filter(x, label_column="bandwidth_variable")
        ]
        not_unstable_filters = [
            lambda x: not unstable_filter(x, label_column="bandwidth_0.13"),
            lambda x: not unstable_filter(x, label_column="bandwidth_variable")
        ]

        for bandwidth_type, cl_column, lens_fil, not_unstable_fil in zip(
            ["fixed", "variable"],
            ["bandwidth_0.13", "bandwidth_variable"],
            lens_filters,
            not_unstable_filters
            ):
            filtered_data = data.groupby(by="objectid").filter(
                not_unstable_fil
            )
            filtered_data = (filtered_data.groupby(by="objectid")
                .filter(lens_fil)
            )

            transformed_data = make_lensing_dataframe(
                filtered_data,
                label_column=cl_column
            )
            transformed_data_index = pd.MultiIndex.from_arrays(
                [
                    [i_batch] * transformed_data.shape[0],
                    [bandwidth_type] * transformed_data.shape[0],
                    transformed_data.index.get_level_values(0),
                    transformed_data.index.get_level_values(1)
                ],
                names=(
                    ["batch_number", "bandwidth_type"] +
                    list(transformed_data.index.names)
                )
            )
            transformed_data.index = transformed_data_index
            transformed_dataframes.append(transformed_data)
        result = pd.concat(transformed_dataframes, axis=0)
        return result

    def run(self, **kwargs):
        """
        Run the task.

        Parameters:
        ----------
        kwargs : dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):

                - batch_range: (tuple, optional, default: (0, 66)): A tuple 
                    specifying the start (inclusive) and stop (inclusive)
                    batch index numbers to process. For example, 
                    (0, 66) will process batches 0 through 66.
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

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(
            f"{self.load_dir}background_results_batch*.parquet"
        )
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}background_results.parquet")

    def get_extract_file_path(self, i_batch):
        """
        Get the extract file path corresponding to batch index.
        """
        result = os.path.join(
            self.extract_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
        )
        return result

    def get_load_file_path(self, i_batch):
        """
        Get the load file path corresponding to batch index
        """
        result = os.path.join(
            self.load_dir, f"background_results_batch{i_batch}.parquet"
        )
        return result
