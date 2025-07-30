"""
This module defines LensLightcurvesTask.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pipeline.etl_task import ETLTask
from microlensing.kde_label import cluster_label_dataframe

class LensLightcurvesTask(ETLTask):
    """
    LensLightcurvesTask
    """
    DEFAULT_BATCH_ARRAY = np.arange(0, 132)
    DEFAULT_TAU_ARRAY = np.arange(0, 49)
    EVENT_DURATIONS = np.geomspace(1e-4, 1e4, num=50)
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": [
                "asemi", 
                "bsemi", 
                "class_star",
                "exptime",
                "filter",
                "flags",
                "fwhm",
                "kron_radius",
                "mag_aper1",
                "mag_aper2",
                "mag_aper4",
                "mag_aper8",
                "mag_auto",
                "magerr_aper1",
                "magerr_aper2",
                "magerr_aper4",
                "magerr_aper8",
                "magerr_auto",
                "mjd",
                "mjd_mid",
                "objectid"
            ]
        }
    }

    def transform(
        self,
        data,
        event_duration,
        sampled_windows
    ):
        """
        Transform the data.

        Parameters:
        ----------
        data : pandas.DataFrame
            The data to transform.
        i_tau : int
            Which tau index.

        Returns:
        ----------
        transformed_data : pandas.DataFrame
            The transformed data.
        """
        sampled_windows = sampled_windows.reset_index()
        object_list = sampled_windows["objectid"].unique()
        filtered_data = data.loc[data["objectid"].isin(object_list)]
        df_merged = sampled_windows.merge(filtered_data, on="objectid")
        g = df_merged.groupby(by=["objectid", "window_number"], as_index=False)
        lensed_lcs = g[df_merged.columns].apply(
            self.lens_apply,
            event_duration
        )

        if not lensed_lcs.empty:
            result = cluster_label_dataframe(
                lensed_lcs,
                groups=["objectid", "window_number", "filter"]
            ).reset_index(drop=True)
        else:
            result = lensed_lcs.assign(cluster_label=[])

        return result

    @staticmethod
    def lens_apply(df, event_duration):
        mag_columns = [
            "mag_aper1",
            "mag_aper2",
            "mag_aper4",
            "mag_aper8",
            "mag_auto"
        ]
        result = df.assign(true_label=np.ones(df.shape[0], dtype=int))
        mask_mjd = (
            (df["mjd_mid"] > df["t_start"].iloc[0]) &
            (df["mjd_mid"] < (df["t_start"].iloc[0] + event_duration))
        )
        result.loc[mask_mjd, mag_columns] += -2.5 * np.log10(2)
        result.loc[mask_mjd, "true_label"] = 0
        return result

    def get_extract_file_path(self, i_batch):
        """
        Get the extract file path.

        Parameters:
        ----------
        i_tau: int
            Which tau index.

        Returns:
        ----------
        extract_file_path : str
            The path to the extract file.
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
        kwargs: dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
        """
        run_kwargs = {}

        if "batch_range" in kwargs:
            batch_range = kwargs.pop("batch_range")
            batch_array = np.arange(batch_range[0], batch_range[1]+1)
        else:
            batch_array = self.DEFAULT_BATCH_ARRAY

        if "tau_range" in kwargs:
            tau_range = kwargs.pop("tau_range")
            tau_array = np.arange(tau_range[0], tau_range[1]+1)
        else:
            tau_array = self.DEFAULT_TAU_ARRAY

        run_kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"].copy()
        sampled_windows_path = kwargs.get("sampled_windows_path", None)

        if sampled_windows_path is None:
            raise ValueError("Argument 'sampled_windows_path' is required.")

        for i_batch in tqdm(batch_array):
            lc_file = self.get_extract_file_path(i_batch)
            data = self.extract(lc_file, **run_kwargs["extract"])

            for i_tau in tqdm(tau_array):
                sampled_windows_file = os.path.join(
                    sampled_windows_path,
                    f"sampled_windows_tau{i_tau}.parquet"
                )

                try:
                    sampled_windows = pd.read_parquet(
                        sampled_windows_file,
                        columns=["t_start"]
                    )
                except FileNotFoundError:
                    print(f"File not found: {sampled_windows_file}. Skipping.")
                    continue

                transformed_data = self.transform(
                    data,
                    self.EVENT_DURATIONS[i_tau],
                    sampled_windows
                )
                load_file_path = self.get_load_file_path(i_batch, i_tau)

                if not transformed_data.empty:
                    # Make a load directory if one doesn't exist
                    os.makedirs(os.path.dirname(load_file_path), exist_ok=True)
                    self.load(transformed_data, load_file_path)
                else:
                    print(f"No data for {load_file_path}. Skipping.")


    def get_load_file_path(self, i_batch, i_tau):
        """
        Get the load file path.

        Parameters:
        ----------
        i_tau: int
            Which tau index.

        Returns:
        ----------
        load_file_path : str
            The path to the load file.
        """
        result = os.path.join(
            self.load_dir,
            f"lensed_lightcurves_batch{i_batch}_duration{i_tau}.parquet"
        )
        return result
