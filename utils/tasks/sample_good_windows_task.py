"""
This module defines SampleGoodWindowsTask.
"""
import os
import numpy as np
import pandas as pd
from utils.tasks.etl_task import ETLTask

class SampleGoodWindowsTask(ETLTask):
    """
    SampleGoodWindowsTask iterates over the event durations and randomly takes
    rows from the good_windows dataframes, randomly chooses a t_start at which
    to begin the lensing event, and writes the resulting dataframe.

    Attributes:

        extract_dir (str): 
            Directory containing the input data files to be processed.

        load_dir (str): 
            Directory where the transformed data files will be written.

    Methods:
    """
    @staticmethod
    def t_start_df(df, n_samples, rng_seed=None):
        """
        Helper function for use with transform. Randomly samples n_samples
        windows from the dataframe and randomly selects a start time
        between 'start' and 'end'.

        Parameters:
        ----------
        df (pandas.DataFrame):
            The dataframe from which to select windows.
        n_samples (int):
            The number of windows to sample.
        rng_seed (
            None, int, array_like[ints], SeedSequence, BitGenerator, 
            Generator, RandomState
        ) optional:
            Seed for the numpy random number generator
        """
        rng = np.random.default_rng(seed=rng_seed)
        s = df.xs("start", level=2)
        e = df.xs("end", level=2)
        time_df = pd.concat([s, e], axis=1)
        time_df.columns = ["start", "end"]

        if time_df.shape[0] >= n_samples:
            result = time_df.sample(n=n_samples)
        else:
            result = time_df

        result["t_start"] = rng.uniform(
            low=result["start"],
            high=result["end"]
        )
        return result

    def transform(self, data, *args, **kwargs):
        """
        Transform the data.

        Parameters:
        ----------
        data (pd.DataFrame):
            The data to transform.

        args:
            Positional arguments passed to the method. Includes:

            i_tau (int): This argument is not used in this implementation.

            version (str): Specifies which version of achromaticity 
                requirements the method uses.

        kwargs (dict):
            Keyword arguments for configuring the task. Used for compatibility
            with ETLTask. This method expects the following key(s):

            summary_table (pandas.DataFrame, required):
                A dataframe of source IDs and lightcurve classifications
                used to filter good windows from only unimodal sources.

            n_samples (int, required):
                The number of windows to sample.

            rng_seed (
                None, int, array_like[ints], SeedSequence, BitGenerator, 
                Generator, RandomState
            ) optional:
                Seed for the numpy random number generator
        """
        version = args[1]
        rng_seed = kwargs.get("rng_seed", None)
        summary_table = kwargs["summary_table"]
        n_samples = kwargs["n_samples"]
        result_data = []

        for bandwidth_type in ["fixed", "variable"]:
            unimodal_ids = summary_table.loc[
                (
                    summary_table[f"{bandwidth_type}_bw_{version}_lc_class"]
                    == "unimodal"
                )
            ].index.to_numpy()
            filtered_data = data.loc[
                data.index.get_level_values(0).isin(unimodal_ids)
            ]

            if not filtered_data.empty:
                t_start_data = self.t_start_df(
                    filtered_data,
                    n_samples,
                    rng_seed=rng_seed
                )
                result_data.append(
                    pd.concat(
                        [t_start_data],
                        keys=[bandwidth_type],
                        names=["bandwidth_type"]
                    )
                )

        if len(result_data) > 0:
            result = pd.concat(result_data)
        else:
            result = pd.DataFrame(
                columns=["start", "end", "t_start"],
                index=pd.MultiIndex.from_arrays(
                    [[], [], []],
                    names=["bandwidth_type", "objectid", "number"]
                )
            )
        return result

    def get_extract_file_path(self, i_tau, version):
        """
        Get the extract file path.

        Parameters:
        ----------
        i_tau (int):
            Which event duration.

        version (str):
            Which version of achromaticity requirements.
        """
        result = os.path.join(
            self.extract_dir,
            f"good_windows_tau{i_tau}_{version}.parquet"
        )
        return result

    def run(self, **kwargs):
        """
        Run the task.

        Parameters:
        ----------
        kwargs (dict):
            Keyword arguments for configuring the task. This method expects
            the following key(s):

            summary_table_path (str, required):
                A string giving the path to the summary table used to filter
                entries corresponding to unimodal lightcurves.

            tau_range (tuple, optional, default=(0, 48)): 
                A tuple specifying the first (inclusive) and last (inclusive)
                event durations to process.

            versions (list, optional, default=['v0', 'v1', 'v2']):
                A list specifying which versions (achromaticity requirements)
                to process.

            n_samples (int, optional, default=100000):
                The number of windows to sample.

            rng_seed (int, optional, default=None):
                The seed for the numpy random number generator.
        """
        summary_table_path = kwargs.get("summary_table_path", None)

        if summary_table_path is None:
            raise ValueError("Argument 'summary_table_path' is required.")

        kwargs["transform"] = {
            "rng_seed": kwargs.pop("rng_seed", None),
            "n_samples": kwargs.pop("n_samples", 100000),
            "summary_table": pd.read_parquet(
                summary_table_path,
                columns=[
                    "fixed_bw_v0_lc_class",
                    "fixed_bw_v1_lc_class",
                    "fixed_bw_v2_lc_class",
                    "variable_bw_v0_lc_class",
                    "variable_bw_v1_lc_class",
                    "variable_bw_v2_lc_class"
                ]
            )
        }
        tau_range = kwargs.get("tau_range", (0, 48))
        tau_array = np.arange(tau_range[0], tau_range[1] + 1)
        versions = kwargs.get("versions", ["v0", "v1", "v2"])
        kwargs["iterables"] = [tau_array, versions]
        super().run(**kwargs)

    def get_load_file_path(self, i_tau, version):
        """
        Get the load file path.

        Parameters:
        ----------
        i_tau (int):
            Which event duration.

        version (str):
            Which version of achromaticity requirements.
        """
        result = os.path.join(
            self.load_dir,
            f"sampled_windows_tau{i_tau}_{version}.parquet"
        )
        return result
