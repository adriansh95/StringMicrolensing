"""
This module defines EventStatsTask. EventStatsTask
iterates over lightcurves grouped in batches. 
It filters out lightcurves that appear bimodal
according to the KDE classification. It groups these
by objectid, and then iterates over every bright sequence
in the lightcurve. For every unique color-band in the bright
sequence it computes the mean, standard error, and
standard deviation (if possible) in that band.
"""
import os
import numpy as np
import pandas as pd
from pipeline.etl_task import ETLTask
from microlensing.analyze_lensing import calculate_event_statistics

class EventStatsTask(ETLTask):
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
    DEFAULT_ITERABLES = (0, 132)
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": [
                "objectid",
                "filter",
                "mag_auto",
                "magerr_auto",
                "mjd",
                "root_2_label"
            ]
        },
        "transform": {
            "mag_column": "mag_auto",
            "magerr_column": "magerr_auto",
            "label_column": "root_2_label"
        }
    }

    @staticmethod
    def filter_func(df, label_column):
        cl = df[label_column].to_numpy()
        result = ((cl == 0).any() & (cl != -1).all())
        return result

    def transform(self, data, i_batch, **kwargs):
        """
        Transform the data.

        Parameters:
        ----------
        data: `pandas.DataFrame`
            The data to transform.
        """
        filtered_data = data.groupby(by="objectid").filter(
            lambda x: self.filter_func(x, kwargs["label_column"])
        )
        result = filtered_data.groupby(by="objectid").apply(
            calculate_event_statistics,
            include_groups=False,
            **kwargs
        )
        result = result.set_index(
            ["event_number", "filter"],
            append=True
        ).reset_index(level=1, drop=True)
        result = pd.concat(
            [result],
            names=["batch_number", "objectid", "event_number", "filter"],
            keys=[i_batch]
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
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
        """
        run_kwargs = {}
        batch_range = kwargs.get(
            "batch_range",
            self.DEFAULT_ITERABLES
        )
        batch_array = np.arange(batch_range[0], batch_range[1]+1, dtype=int)
        run_kwargs["iterables"] = [batch_array]
        run_kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"]
        run_kwargs["transform"] = self.DEFAULT_RUN_KWARGS["transform"]

        if "mag_column" in kwargs:
            run_kwargs["extract"]["columns"][2] = kwargs["mag_column"]
            run_kwargs["transform"]["mag_column"] = kwargs["mag_column"]

        if "magerr_column" in kwargs:
            run_kwargs["extract"]["columns"][3] = kwargs["magerr_column"]
            run_kwargs["transform"]["magerr_column"] = kwargs["magerr_column"]

        if "label_column" in kwargs:
            run_kwargs["extract"]["columns"][5] = kwargs["label_column"]
            run_kwargs["transform"]["label_column"] = kwargs["label_column"]

        super().run(**run_kwargs)

    def get_load_file_path(self, i_batch):
        """
        Get the load file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.load_dir,
            f"event_stats_batch{i_batch}.parquet"
        )
        return result

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(f"{self.load_dir}event_stats_batch*.parquet")
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}event_stats.parquet")
