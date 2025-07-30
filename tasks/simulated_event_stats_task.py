"""
This module defines SimulatedEventStatsTask. SimulatedEventStatsTask
iterates over lightcurves grouped in batches. 
It filters out lightcurves that appear bimodal
according to the KDE classification. It groups these
by objectid, and then iterates over every bright sequence
in the lightcurve. For every unique color-band in the bright
sequence it computes the mean, standard error, and
standard deviation (if possible) in that band.
"""
import os
import glob
import numpy as np
import pandas as pd
from pipeline.etl_task import ETLTask
from microlensing.analyze_lensing import calculate_event_statistics

class SimulatedEventStatsTask(ETLTask):
    """
    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.

    Methods:
        get_extract_file_path(*keys):
            Returns the file_path to process given the keys.
        get_load_file_path(*keys):
            Returns the file_path to load given the keys.
        transform(data, *keys):
            Transform the data.
    """
    DEFAULT_ITERABLES = (0, 48)
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": [
                "objectid",
                "window_number",
                "filter",
                "mag_auto",
                "magerr_auto",
                "mjd",
                "true_label"
            ]
        }
    }

    @staticmethod
    def clean_data(df):
        mask_baseline = df["cluster_label"].astype(bool).to_numpy()
        bright_filters = df.loc[~mask_baseline, "filter"].unique()
        baseline_filters = df.loc[mask_baseline, "filter"].unique()
        result = np.isin(bright_filters, baseline_filters).all()
        return result

    @staticmethod
    def filter_func(df):
        cl = df["cluster_label"].to_numpy()
        result = ((cl == 0).any() & (cl != -1).all())
        return result

    def transform(self, data, i_duration):
        """
        Transform the data.

        Parameters:
        ----------
        data: `pandas.DataFrame`
            The data to transform.
        """
        data = data.rename(columns={"true_label": "cluster_label"})
        data = (
            data
            .groupby(by=["objectid", "window_number"])
            .filter(self.clean_data)
        )
        filtered_data = data.groupby(
            by=["objectid", "window_number"]
        ).filter(self.filter_func)
        result = filtered_data.groupby(
            by=["objectid", "window_number"]
        ).apply(calculate_event_statistics, include_groups=False)
        result = result.set_index(
            ["event_number", "filter"],
            append=True
        ).reset_index(level=2, drop=True)
        result = pd.concat(
            [result],
            names=[
                "duration_index",
                "objectid",
                "sim_number",
                "event_number",
                "filter"
            ],
            keys=[i_duration]
        )
        return result

    def get_extract_file_path(self, i_duration):
        """
        Get the extract file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.extract_dir,
            f"lensed_lightcurves_duration{i_duration}.parquet"
        )
        return result

    def run(self, **kwargs):
        """
        Run the task.

        Parameters:
        ----------
        kwargs : dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
        """
        duration_range = kwargs.get(
            "duration_range",
            self.DEFAULT_ITERABLES
        )
        duration_array = np.arange(
            duration_range[0],
            duration_range[1]+1,
            dtype=int
        )
        kwargs["iterables"] = [duration_array]
        kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"]
        super().run(**kwargs)

    def get_load_file_path(self, i_duration):
        """
        Get the load file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.load_dir,
            f"event_stats_duration{i_duration}.parquet"
        )
        return result

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(f"{self.load_dir}event_stats_duration*.parquet")
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}event_stats.parquet")
