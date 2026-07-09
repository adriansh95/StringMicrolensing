"""
This module defines GoodWindowsTask.
"""
import os
from itertools import product
import pandas as pd
import numpy as np
from microlensing.lc_scanner import LcScanner
from microlensing.helpers import filter_map
from pipeline.etl_task import ETLTask
from tasks.task_helpers import load_plugin_from_path

class GoodWindowsTask(ETLTask):
    """
    GoodWindowsTask uses microlensing.lc_scanner.LcScanner to scan 
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
    DEFAULT_ITERABLES = [np.arange(0, 132)]
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": [
                "objectid",
                "mjd",
                "exptime",
                "filter",
            ]
        },
        "transform": {
            "duration_bin_bounds": [1e-4, 1e4],
            "n_duration_bins": 50,
            "bound_both_sides": True,
            "sample_frac": 0.01
        }
    }

    @staticmethod
    def _good_windows_df(data, scanner):
        """
        Helper method for self.transform.

        Parameters:
        ----------
        data : `pandas.DataFrame`
            The data to transform.
        scanner : `microlensing.lc_scanner.LcScanner`

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
                scanner : `microlensing.lc_scanner.LcScanner`
                version : `str`
                    Which achromaticity version is being used.

        Returns:
        ----------
        transformed_data : `pandas.DataFrame`
            The transformed data.
        """
        duration_bins = np.geomspace(
            *kwargs["duration_bin_bounds"],
            num=kwargs["n_duration_bins"]
        )
        durations = (duration_bins[1:] + duration_bins[:-1]) / 2
        scanner = LcScanner(
            durations,
            kwargs["scanner_plugin_func"],
            bound_both_sides=kwargs["bound_both_sides"]
        )
        data.sort_values(by=["objectid", "mjd"], inplace=True)
        data["filter_index"] = data["filter"].apply(filter_map)
        objects = pd.Series(data["objectid"].unique())
        sampled_objects = objects.sample(frac=kwargs["sample_frac"])
        data = data.loc[data["objectid"].isin(sampled_objects)]
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
        run_kwargs = {}

        if "scanner_plugin_file" not in kwargs:
            raise ValueError(
                "'scanner_plugin_file' is a required keyword argument."
            )

        if "batch_range" in kwargs:
            batch_range = kwargs.pop("batch_range")
            run_kwargs["iterables"] = [np.arange(batch_range[0], batch_range[1]+1)]
        else:
            run_kwargs["iterables"] = self.DEFAULT_ITERABLES

        run_kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"].copy()
        run_kwargs["transform"] = self.DEFAULT_RUN_KWARGS["transform"].copy()
        run_kwargs["transform"].update(
            {
                k: kwargs[k]
                for k in self.DEFAULT_RUN_KWARGS["transform"]
                if k in kwargs
            }
        )
        run_kwargs["transform"].update(
            {
                "scanner_plugin_func": load_plugin_from_path(
                    kwargs.pop("scanner_plugin_file")
                )
            }
        )
        super().run(**run_kwargs)

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
