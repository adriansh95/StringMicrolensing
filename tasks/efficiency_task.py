"""
This module contains the EfficiencyTask class which iterates over the
lensed lightcurves and computes some quantities useful for calculating
efficiency and quality.
"""
import os
import glob
import pandas as pd
import numpy as np
from microlensing.filtering import (
    good_photometry,
    flags_filter,
    morphology_filter,
    area_filter,
    class_star_filter,
    lens_filter,
    mahalanobis_distance_ellipse,
    good_detection_filter,
    can_recover1,
    can_recover3,
    can_recover1_alt,
    can_recover2
)
from microlensing.kde_label import cluster_label_dataframe
from pipeline.etl_task import ETLTask

class EfficiencyTask(ETLTask):
    """
    This class reads in lightcurves with synthetic lensing events
    from parquet files and counts how many pass detection criteria,
    how many samples are correctly labeled, and how many times an event
    is "split" by a mislabelled sample.

    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.
    Methods:
        get_extract_file_path(version, i_tau, bandwidth_type):
            Returns the file_path to process given the version, event duration
            index (i_tau), and bandwidth type..
        get_load_file_path(version, i_tau, bandwidth_type):
            Returns the file_path to load given the version, event duration
            index (i_tau), and bandwidth type..
        transform(data, version, i_tau, bandwidth_type):
            Computes the weighted std per band, rms error per band, and 
            lc_class for each source for given achromaticity version,
            event duration index (i_tau), and bandwidth type..
        run(): 
            Executes the ETL process for each file, applying extract,
            transform, and load sequentially.
    """
    DEFAULT_ITERABLES = [np.arange(0, 49)]

    @staticmethod
    def footprint_area_z_score(df, label_column="cluster_label"):
        m = df[label_column].astype(bool)
        result = (
            (df["footprint_area"] - df.loc[m, "footprint_area"].mean())
            / df.loc[m, "footprint_area"].std()
        )
        return result

    @staticmethod
    def clean_data(data):
        data = data.groupby(
            by=["objectid", "window_number", "filter"],
            as_index=False
        ).filter(lambda x: len(x) > 1)

        data = data.groupby(
            by=["objectid", "window_number", "filter"],
            as_index=False
        ).filter(
            lambda x: x.loc[x["true_label"].astype(bool)].shape[0] > 2
        )
        data = data.groupby(
            by=["objectid", "window_number"],
            as_index=False
        ).filter(
            lambda x: (
                lens_filter(
                    x,
                    min_per_filter=1,
                    n_filters_req=2,
                    label_column="true_label",
                    factor_of_two=False
                )
            )
        )
        result = data.reset_index()
        return result

    def compute_filtering_quantities(self, data):
        data = (
            data.groupby(
                by=["objectid", "window_number"],
                as_index=False
            )[data.columns].apply(mahalanobis_distance_ellipse)
        ).reset_index(level=0, drop=True)

        data = data.assign(
            footprint_area=(
                lambda x: (
                    np.pi * x.asemi * x.bsemi
                    * (2.5 * x.kron_radius)**2
                )
            )
        )

        data["footprint_area_z_score"] = data.groupby(
            by=["objectid", "window_number"],
            as_index=False
        ).apply(self.footprint_area_z_score).reset_index(level=0, drop=True)

        data = cluster_label_dataframe(
            data,
            groups=["objectid", "window_number", "filter"],
            mag_column="mag_aper1",
            magerr_column="magerr_aper1",
            output_label_column="aper1_label"
        )
        result = cluster_label_dataframe(
            data,
            groups=["objectid", "window_number", "filter"],
            mag_column="mag_aper2",
            magerr_column="magerr_aper2",
            output_label_column="aper2_label"
        )
        return result

    @staticmethod
    def n_filtered(data, filter_func):
        flag_filtered = (
            data.groupby(
                by=["objectid", "window_number"]
            ).filter(filter_func)
        )
        result = flag_filtered.groupby(by=["objectid", "window_number"]).ngroups
        return result

    def transform(self, data, i_tau):
        data = self.clean_data(data)

        if data.empty:
            result = data
            return result

        data = self.compute_filtering_quantities(data)
        first_pass = (
            data
            .groupby(by=["objectid", "window_number"])
            .filter(
                lambda x: (
                    lens_filter(
                        x,
                        min_per_filter=1,
                        n_filters_req=2
                    )
                )
            )
        )

        sus_photometry = (
            first_pass
            .groupby(by=["objectid", "window_number"])
            .filter(lambda x: ~good_photometry(x))
        )
        g_sus = sus_photometry.groupby(by=["objectid", "window_number"])
        n_sus = g_sus.ngroups
        recovered = g_sus.filter(can_recover3)

#        recovered1_flag0 = g_sus.filter(
#            lambda x: can_recover1(x, flag_cutoff=0)
#        )
#        recovered1_flag1 = g_sus.filter(
#            lambda x: can_recover1(x, flag_cutoff=1)
#        )
#        recovered1_alt = g_sus.filter(can_recover1_alt)
#        recovered2 = g_sus.filter(can_recover2)

        result_data = {
            "n_events": [
                data.groupby(by=["objectid", "window_number"]).ngroups
            ],
            "n_first_pass": [
                first_pass.groupby(by=["objectid", "window_number"]).ngroups
            ],
#            "n_flag0_filtered": [
#                self.n_filtered(
#                    first_pass,
#                    lambda x: ~flags_filter(x, flag_cutoff=0)
#                )
#            ],
#            "n_flag1_filtered": [
#                self.n_filtered(
#                    first_pass,
#                    lambda x: ~flags_filter(x, flag_cutoff=1)
#                )
#            ],
#            "n_flag_alt_filtered": [
#                self.n_filtered(
#                    first_pass,
#                    lambda x: ~flags_filter(x, flag_cutoff=1)
#                )
#            ],
#            "n_shape_filtered": [
#                self.n_filtered(
#                    first_pass,
#                    lambda x: ~(
#                        morphology_filter(x) &
#                        area_filter(x) &
#                        class_star_filter(x)
#                    )
#                )
#            ],
            "n_sus_photometry": n_sus,
#            "n_recovered1_flag0": (
#                recovered1_flag0
#                .groupby(by=["objectid", "window_number"])
#                .ngroups
#            ),
#            "n_recovered1_flag1": (
#                recovered1_flag1
#                .groupby(by=["objectid", "window_number"])
#                .ngroups
#            ),
#            "n_recovered1_alt": (
#                recovered1_flag1
#                .groupby(by=["objectid", "window_number"])
#                .ngroups
#            ),
            "n_recovered": (
                recovered
                .groupby(by=["objectid", "window_number"])
                .ngroups
            ),
#             "n_recovered2": (
#                recovered2
#                .groupby(by=["objectid", "window_number"])
#                .ngroups
#            ),
            "n_good_det": [
                self.n_filtered(
                    data,
                    good_detection_filter
                )
            ]
        }
        baseline_mask = first_pass["cluster_label"].astype(bool)
        result_data["n_baseline"] = first_pass.loc[baseline_mask].shape[0]
        result_data["n_bright"] = first_pass.loc[~baseline_mask].shape[0]
        baseline_flag_counts = (
            first_pass.loc[baseline_mask, "flags"].value_counts()
        )
        baseline_flag_dict = {
            f"baseline_n_flag_{flag_val}": [
                baseline_flag_counts.loc[flag_val]
            ] for flag_val in baseline_flag_counts.index
        }
        bright_flag_counts = (
            first_pass.loc[~baseline_mask, "flags"].value_counts()
        )
        bright_flag_dict = {
            f"bright_n_flag_{flag_val}": [
                bright_flag_counts.loc[flag_val]
            ] for flag_val in bright_flag_counts.index
        }
        result_data.update(baseline_flag_dict)
        result_data.update(bright_flag_dict)
        result = pd.DataFrame(
            data=result_data,
            index=[i_tau]
        )
        result.index.name = "duration_index"

        #g = data.groupby(by=["objectid", "number"], sort=False)
        #lc_class_df = g.apply(
        #    lambda x: lightcurve_classifier(x, **self.config[version])
        #)
        #n_detections = sum(lc_class_df == "background")
        #n_injections = g.ngroups
        #n_correctly_labeled = (
        #    data["cluster_label"] == data["true_label"]
        #).sum()
        #n_samples = data.shape[0]
        #n_windows = g["cluster_label"].apply(
        #    lambda x: len(get_bounding_idxs(x.to_numpy()))
        #)
        #n_splits = (n_windows - 1).sum()
        #data = {
        #    "n_detections": [n_detections],
        #    "n_injections": [n_injections],
        #    "n_correctly_labeled": [n_correctly_labeled],
        #    "n_samples": [n_samples],
        #    "n_splits": [n_splits]
        #}
        #idx = pd.MultiIndex.from_tuples(
        #    [(version, i_tau, bandwidth_type)],
        #    names=["version", "tau_index", "bandwidth_type"]
        #)
        #result = pd.DataFrame(data=data, index=idx)
        return result

    def run(self, **kwargs):
        """
        Run the task. It accepts the following keyword arguments:
            tau_range: (tuple, optional, default: (0, 48)): A tuple 
                specifying the range of tau indices to process. The first
                element specifies the starting index (inclusive) and the second
                specifies the last (inclusive).
        """
        run_kwargs = {}

        if "tau_range" in kwargs:
            tau_range = kwargs["tau_range"]
            run_kwargs["iterables"] = [np.arange(tau_range[0], tau_range[1]+1)]
        else:
            run_kwargs["iterables"] = self.DEFAULT_ITERABLES

        super().run(**run_kwargs)

    def concat_results(self):
        """
        Concatenate the results from ETL into a single dataframe.
        """
        df_files = glob.glob(f"{self.load_dir}efficiency_results_tau*.parquet")
        dfs = [pd.read_parquet(f) for f in df_files]
        result = pd.concat(dfs, axis=0)
        result.sort_index(inplace=True)
        result.to_parquet(f"{self.load_dir}efficiency_results.parquet")

    def get_extract_file_path(self, i_tau):
        """
        Get the extract file path corresponding to the given
        event duration (specified indirectly by i_tau)
        """
        result = os.path.join(
            self.extract_dir,
            f"lensed_lightcurves_duration{i_tau}.parquet"
        )
        return result

    def get_load_file_path(self, i_tau):
        """
        Get the extract file path corresponding to the given
        event duration (specified indirectly by i_tau).
        """
        result = os.path.join(
            self.load_dir, f"efficiency_results_tau{i_tau}.parquet"
        )
        return result
