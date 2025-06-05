"""
This module defines EffectiveMonitoringTimeTask.
"""
import os
import glob
import numpy as np
import pandas as pd
from pipeline.etl_task import ETLTask

class EffectiveMonitoringTimeTask(ETLTask):
    """
    EffectiveMonitoringTimeTask takes the difference between the
    end and start of every 'good' lensing window ('good' depends
    on which version (achromaticity requirements) is used) and
    sums these differences to compute the total effective
    monitoring time.
    """
    DEFAULT_ITERABLES = (0, 66)

    def transform(self, data, i_batch, **kwargs):
        """
        Transform the data.

        Parameters:
        ----------
        data (pandas.DataFrame):
            The data to transform.
        i_batch : int
            Which batch number to process.
        """
        sampled_ids = kwargs["sampled_ids"]
        mask = data.index.get_level_values(1).isin(sampled_ids)
        filtered_data = data.loc[mask]
        time_diffs = (
            filtered_data.xs("end", level=3) -
            filtered_data.xs("start", level=3)
        )
        result = pd.concat(
            [time_diffs.groupby(level=1).agg("sum")],
            keys=[i_batch],
            names=["batch_number"]
        )
        return result

    def get_extract_file_path(self, i_batch):
        """
        Get the extract file path.

        Parameters:
        ----------
        i_batch (int):
            Which batch number for which to get data.
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
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
                - sampled_ids_file (required) : string
                    Path pointing to sampled_objects.parquet, the file
                    containing the object ids, ra, and dec of the random
                    subset of sources.

                - batch_range (optional) : `tuple of (int, int)`
                    A tuple specifying the start (inclusive) and stop 
                    (inclusive) batch index numbers to process. For example, 
                    the default value of (0, 66) will process 
                    batches 0 through 66.
        """
        batch_range = kwargs.get(
            "batch_range",
            self.DEFAULT_ITERABLES
        )
        batch_array = np.arange(batch_range[0], batch_range[1]+1, dtype=int)
        kwargs["iterables"] = [batch_array]

        if "sampled_ids_file" not in kwargs:
            raise ValueError("'sampled_ids_file' is a required keyword argument.")

        kwargs["transform"] = {
            "sampled_ids": pd.read_parquet(
                kwargs.pop("sampled_ids_file"),
                columns=["id"]
            )["id"].to_numpy()
        }
        super().run(**kwargs)

    def get_load_file_path(self, i_batch):
        """
        Get the load file path.

        Parameters:
        ----------
        i_batch (int):
            Which batch number processed.
        """
        result = os.path.join(
            self.load_dir,
            f"effective_monitoring_time_batch{i_batch}.parquet"
        )
        return result

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(
            f"{self.load_dir}effective_monitoring_time_batch*.parquet"
        )
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}effective_monitoring_time.parquet")
