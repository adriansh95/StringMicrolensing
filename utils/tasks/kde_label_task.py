"""
This module contains the KDELabelTask class.
"""
import os
import numpy as np
from utils.tasks.etl_task import ETLTask
from utils.kde_label import cluster_label_dataframe

class KDELabelTask(ETLTask):
    """
    This class reads in lightcurves from parquet files, cleans the data
    by removing rows that filters that are measured fewer than 3 times
    per source (Not enough to establish baseline and bright samples),
    then cluster labels the samples using a variable bandwidth and
    fixed bandwidth of 130 mmag, adds two new columns to the data, and writes
    the transformed data to load_dir.

    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.

    Methods:
        get_file(i_batch): Returns the file to processe given the batch number.
        extract(file): Reads data from a given input file and returns it
                       as a DataFrame.
        transform(data): Applies the filtering condition and adds new columns
                       to the DataFrame.
        load(data, i_batch): Writes the transformed DataFrame to the specified
                             output directory.
        run(): Executes the ETL process for each file, applying extract,
               transform, and load sequentially.
    """

    def transform(self, data, *args):
        """Clean the data and add the cluster labels"""
        data.sort_values(by=["objectid", "mjd"], inplace=True)
        g = data.groupby(by="objectid")

        # Some sources have "simultaneous" measurements. Filter those out.
        data = g.filter(
            lambda x: (np.diff(x["mjd"].to_numpy()) > 0).all()
        )
        bandwidths = ["variable", 0.13]
        col_names = [f"bandwidth_{bw}" for bw in bandwidths]
        g = data.groupby(
            by=["objectid", "filter"],
            group_keys=False,
            sort=False
        )
        result = g.filter(lambda x: len(x) > 2)
        cl_data = np.zeros((result.shape[0], len(bandwidths)), dtype=int)
        cl = "cluster_label"

        for i, bw in enumerate(bandwidths):
            cl_data[:, i] = cluster_label_dataframe(result, bandwidth=bw)[cl]

        result[col_names] = cl_data
        return result

    def run(self, **kwargs):
        """
        Run the task. It accepts the following keyword argument:
            batch_range : `tuple of (int, int)`
                A tuple specifying the range of batch indices to process. 
                Defaults to (0, 66). The first element specifies the starting
                index (inclusive) and the second specifies the last (inclusive).
        """
        batch_range = kwargs.pop("batch_range", (0, 66))
        first_batch = batch_range[0]
        last_batch = batch_range[1]
        batch_array = np.arange(first_batch, last_batch+1)
        kwargs["iterables"] = [batch_array]
        super().run(**kwargs)

    def get_extract_file_path(self, *args):
        """Get the extract file path corresponding to i_batch"""
        i_batch, = args
        result = os.path.join(
            self.extract_dir, f"lightcurves_batch{i_batch}.parquet"
            )
        return result

    def get_load_file_path(self, *args):
        """Get the load file path corresponding to i_batch"""
        i_batch, = args
        result = os.path.join(
            self.load_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
            )
        return result
