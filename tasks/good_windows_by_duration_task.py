"""
This module defines GoodWindowsByDurationTask.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipeline.etl_task import ETLTask

class GoodWindowsByDurationTask(ETLTask):
    """
    GoodWindowsByDurationTask runs after GoodWindowsTask.
    The dataframes resulting from GoodWindowsTask are more
    useful if they are collated by column (event duration)
    instead of by batch. This task iterates over the columns
    and concatenates over all batches.

    Attributes:
        extract_dir: `str`
            Directory containing the input data files to be processed.
        load_dir: `str`
            Directory where the transformed data files will be written.
        config_paths: `dict`
            Dictionary with key 'py_path'. This points the task to a
            python file with parameters to configure the analysis.

    Methods:
        get_extract_file_path(i_batch):
            Returns the file_path to process given the batch number.
        get_load_file_path(i_batch):
            Returns the file_path to load given the batch_number.
        transform(data, *keys):
            Transform the data.
    """

    def transform(self, data):
        """
        Transform the data.

        Parameters:
        ----------
        data : `list of [pandas.DataFrame]`
            The data to transform.

        Returns:
        ----------
        transformed_data : `pandas.DataFrame`
            The transformed data.
        """
        result = pd.concat(
            [df.dropna() for df in data]
        )
        return result

    def get_extract_file_path(self, i_batch):
        """
        Get the extract file path.

        Parameters:
        ----------
        i_batch: `int`
            Batch number for which to return the file path
        """
        result = os.path.join(
            self.extract_dir,
            f"good_windows_batch{i_batch}.parquet"
        )
        return result

    def run(self, **kwargs):
        """
        Run the task.

        Parameters:
        ----------
        kwargs : dict
            Keyword arguments for configuring the task. This method expects
            the following key(s):
                tau_range : `tuple of (int, int)`
                    Which event durations to process. Indicates start
                    and stop, both inclusive.
        """
        run_kwargs = {}
        # size - 1 since endpoints are both inclusive
        tau_array = np.arange(
            *kwargs.pop(
                "tau_range",
                (0, 49)
            )
        )
        run_kwargs["iterables"] = np.arange(
            *kwargs.pop(
                "batch_range",
                (0, 133)
            )
        )

        for i_tau in tqdm(tau_array):
            run_kwargs["extract"] = {"columns": [f"tau_{i_tau}"]}
            extract_file_paths = [
                self.get_extract_file_path(i_batch)
                for i_batch in run_kwargs["iterables"]
            ]
            data_list = [
                self.extract(f, **run_kwargs["extract"])
                for f in extract_file_paths
            ]
            transformed_data = self.transform(data_list)
            load_file_path = self.get_load_file_path(i_tau)

            if not transformed_data.empty:
                self.load(transformed_data, load_file_path)
            else:
                print(f"No data for {load_file_path}. Skipping.")

    def get_load_file_path(self, i_tau):
        """
        Get the load file path.

        Parameters:
        ----------
        i_tau: `int`
            Which event duration is being processed.
        """
        result = os.path.join(
            self.load_dir,
            f"good_windows_tau{i_tau}.parquet"
        )
        return result
