"""
This module contains the KDELabelTask class.
"""
import os
import numpy as np
import pandas as pd
from microlensing.kde_label import cluster_label_dataframe
from pipeline.etl_task import ETLTask

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

    DEFAULT_ITERABLES = [np.arange(0, 133)]

    def transform(self, data, i_batch):
        """Clean the data and add the cluster labels"""
        data.sort_values(by=["objectid", "mjd"], inplace=True)
        g = data.groupby(
            by="objectid",
            group_keys=False,
            sort=False
        )
        # Some sources have "simultaneous" measurements. Filter those out.
        data = g.filter(
            lambda x: (np.diff(x["mjd"].to_numpy()) > 0).all()
        )
        bandwidth_funcs = [
            lambda x: np.sqrt(np.mean(x**2)),
            lambda x: np.sqrt(2 * np.mean(x**2)),
            lambda x: np.sqrt(9 * np.mean(x**2) / 2)
        ]
        g = data.groupby(
            by=["objectid", "filter"],
            group_keys=False,
            sort=False
        )
        result = g.filter(lambda x: len(x) > 2)
        cl_data = np.zeros((result.shape[0], len(bandwidth_funcs)), dtype=int)
        label_columns = ["root_2_label", "2_label", "3_label"]

        for i, (bw_func, label) in enumerate(
                zip(bandwidth_funcs, label_columns)
            ):
            cl_data[:, i] = cluster_label_dataframe(
                result,
                bandwidth_func=bw_func,
                output_label_column=label
            )[label]

        result[label_columns] = cl_data
        result = pd.concat([result], keys=[i_batch], names=["batch_number"])
        return result

    def run(self, **kwargs):
        """
        Run the task. It accepts the following keyword argument:
            batch_range : `tuple of (int, int)`
                A tuple specifying the range of batch indices to process. 
                Defaults to (0, 66). The first element specifies the starting
                index (inclusive) and the second specifies the last (inclusive).
        """
        run_kwargs = {}

        if "batch_range" in kwargs:
            run_kwargs["iterables"] = [
                np.arange(
                    kwargs["batch_range"][0],
                    kwargs["batch_range"][1]+1
                )
            ]
        else:
            run_kwargs["iterables"] = self.DEFAULT_ITERABLES
        super().run(**run_kwargs)

    def get_extract_file_path(self, i_batch):
        """Get the extract file path corresponding to i_batch"""
        result = os.path.join(
            self.extract_dir, f"lightcurves_batch{i_batch}.parquet"
            )
        return result

    def get_load_file_path(self, i_batch):
        """Get the load file path corresponding to i_batch"""
        result = os.path.join(
            self.load_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
            )
        return result
