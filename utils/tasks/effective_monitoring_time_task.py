"""
This module defines EffectiveMonitoringTimeTask.
"""
import os
import numpy as np
import pandas as pd
import glob
from utils.tasks.etl_task import ETLTask

class EffectiveMonitoringTimeTask(ETLTask):
    """
    EffectiveMonitoringTimeTask takes the difference between the
    end and start of every 'good' lensing window ('good' depends
    on which version (achromaticity requirements) is used) and
    sums these differences to compute the total effective
    monitoring time.
    """
    def transform(self, data, i_batch):
        """
        Transform the data.

        Parameters:
        ----------
        data (pandas.DataFrame):
            The data to transform.
        """
        time_diffs = data.xs("end", level=3) - data.xs("start", level=3)
        result = time_diffs.sum(axis=0).to_frame().T
        result_idx = pd.Index(data=[i_batch], name="batch_number")
        result.index = result_idx
        return result

    def get_extract_file_path(self, i_batch):
        """
        Get the extract file path.

        Parameters:
        ----------
        i_batch (int):
            Which batch number for which to get data.
        version: (str):
            Which version for which to get data.
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
                - batch_range (optional) : `tuple of (int, int)`
                    A tuple specifying the start (inclusive) and stop 
                    (inclusive) batch index numbers to process. For example, 
                    the default value of (0, 66) will process 
                    batches 0 through 66.
        """
        batch_range = kwargs.get("batch_range", (0, 66))
        batch_array = np.arange(batch_range[0], batch_range[1]+1, dtype=int)
        kwargs["iterables"] = [batch_array]
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
