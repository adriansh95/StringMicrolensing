"""
This module defines GoodWindowsTask.
"""
import os
from itertools import product
import pandas as pd
import numpy as np
from utils.tasks.etl_task import ETLTask
from utils.lc_scanner import LcScanner
from utils.helpers import filter_map
from config.config_loader import load_config

class GoodWindowsTask(ETLTask):
    """
    GoodWindowsTask uses utils.lc_scanner.LcScanner to scan 
    lightcurves and record all the time intervals during 
    which a lensing event could start and lens a sufficient 
    subset of the samples.

    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.

    Methods:
        get_extract_file_path(i_batch):
            Returns the file_path to process given the batch number.
        get_load_file_path(i_batch):
            Returns the file_path to load given the batch_number.
        transform(data, *keys):
            Transform the data.
    """
    def __init__(self, extract_dir, load_dir,  config_paths):
        super().__init__(extract_dir, load_dir)
        self.config = load_config(
            yaml_path=config_paths["yaml_path"],
            py_path=config_paths["py_path"]
        )

    @staticmethod
    def _good_windows_df(data, scanner):
        """
        Helper method for self.transform.

        Parameters:
        ----------
        data : `pandas.DataFrame`
            The data to transform.
        scanner : `utils.lc_scanner.LcScanner`

        Returns:
        ----------
        good_windows_df : `pandas.DataFrame`
            The result from scanner formatted into a
            pandas DataFrame.
        """
        good_windows = scanner.record_windows(data)
        window_idx = list(range(good_windows[::2].shape[0]))
        column_names = [f"tau_{i}" for i in range(good_windows.shape[1])]
        idx = pd.MultiIndex.from_tuples(
            list(product(window_idx, ["start", "end"])),
            names=["window_number", "boundary"]
        )
        result = pd.DataFrame(
            data=good_windows,
            index=idx,
            columns=column_names
        )
        return result

    def transform(self, data, i_batch, **kwargs):
        """
        Transform the data.

        Parameters:
        ----------
        data : `pandas.DataFrame`
            The data to transform.
        i_batch : `int`
            Which batch is being processed.
        kwargs : `dict`
            Keyword arguments. This method expects the 
            following keyword argument(s):
                scanner : `utils.lc_scanner.LcScanner`
                version : `str`
                    Which achromaticity version is being used.

        Returns:
        ----------
        transformed_data : `pandas.DataFrame`
            The transformed data.
        """
        scanner = kwargs["scanner"]
        data.sort_values(by=["objectid", "mjd"], inplace=True)
        data["filter_index"] = data["filter"].apply(filter_map)
        g = data.groupby(by="objectid")
        transformed_data = g.apply(
            self._good_windows_df, scanner
        )
        result = pd.concat(
            [transformed_data], keys=[i_batch], names=["batch_number"]
        )
        return result

    def get_extract_file_path(self, i_batch):
        """
        Get the extract file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.extract_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
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
                batch_range : `tuple of (int, int)`
                    Which batch numbers to process. Indicates start
                    and stop, both inclusive.
                version : `str`
                    Which version of achromaticity requirements to run
                bound_both_sides : `bool`
                    Configuring argument for the LcScanner. Determines
                    whether or not hypothetical events must be bounded
                    on both sides in order to be considered a 'good window.'
        """
        batch_range = kwargs.pop("batch_range", (0, 66))
        version = kwargs.pop("version")
        kwargs["iterables"] = [np.arange(batch_range[0], batch_range[1]+1)]
        kwargs["extract"] = {
            "columns": [
                "objectid",
                "mjd",
                "exptime",
                "filter",
            ]
        }
        scanner = LcScanner(
            self.config["taus"],
            n_filters_req=self.config["scanner"][version]["unique_filters"],
            min_per_filter=(
                self.config["scanner"][version]["samples_per_filter"]
            ),
            bound_both_sides = kwargs.pop("bound_both_sides")
        )
        kwargs["transform"] = {
            "version": version,
            "scanner": scanner
        }
        super().run(**kwargs)

    def get_load_file_path(self, i_batch):
        """
        Get the load file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.load_dir,
            f"good_windows_batch{i_batch}.parquet"
        )
        return result
