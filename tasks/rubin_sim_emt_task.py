"""
This module defines RubinSimEMTTask.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from rubin_sim import maf
from microlensing.rubin_sim_utils import (
    EffectiveMonitoringTimeMetric
)
from pipeline.etl_task import ETLTask
from tasks.task_helpers import load_plugin_from_path

class RubinSimEMTTask(ETLTask):
    """
    RubinSimEMTTask takes a pandas.DataFrame of objectid, ra, dec,
    a rubin_sim survey .db file, and evaluates EffectiveMonitoringTimeMetric
    on those positions. 
    """
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": ["id", "ra", "dec"]
        },
        "transform": {
            "duration_bin_bounds": [1e-4, 1e4],
            "n_duration_bins": 50,
            "bounded": True
        }
    }

    def transform(self, data, *args, **kwargs):
        """
        Transform the data.

        Parameters:
        ----------
        data (pandas.DataFrame):
            The data to transform.
        *args:
            Not used, but required to maintain compatibility
            with base class interface
        """
        opsim_db_fname = kwargs["sim_db_file"]
        run_name = Path(opsim_db_fname).stem
        n_duration_bins = kwargs["n_duration_bins"]
        duration_bins = np.geomspace(
            *kwargs["duration_bin_bounds"],
            num=n_duration_bins
        )
        durations = (duration_bins[1:] + duration_bins[:-1]) / 2
        slicer = maf.UserPointsSlicer(
            data["ra"].to_numpy(),
            data["dec"].to_numpy()
        )
        slicer.slice_points["count"] = np.ones(data.shape[0])
        metric = EffectiveMonitoringTimeMetric(
            durations,
            kwargs["scanner_plugin_func"],
            bounded=kwargs["bounded"]
        )
        bundle = maf.MetricBundle(
            metric,
            slicer,
            "",
            run_name=run_name
        )
        group = maf.MetricBundleGroup(
            [bundle],
            opsim_db_fname,
            out_dir=self.load_dir
        )
        group.run_all()
        result_data = group.bundle_dict[
            (
                f"{run_name}_EffectiveMonitoringTimeMetric_USER"
            ).replace(".", "_")
        ].metric_values
        result = pd.DataFrame(
            data=np.vstack(result_data.data[~result_data.mask]),
            index=data.loc[~result_data.mask, "id"].to_numpy(),
            columns=[f"duration_{i}" for i in range(n_duration_bins - 1)]
        )
        result.index.name = "objectid"
        return result

    def get_extract_file_path(self, *args):
        """
        Get the extract file path.

        Parameters:
        ----------
        *args:
            Not used, but required to maintain compatibility with
            base class.
        """
        result = os.path.join(
            self.extract_dir,
            "sampled_objects.parquet"
        )
        return result

    def run(self, **user_kwargs):
        """
        Run the task.

        Parameters:
        ----------
        user_kwargs : dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
                - sim_db_file (required) : string
                    Path pointing to a rubin_sim .db file.

                - scanner_plugin_file (required) : string
                    Path pointing to a plugin for LcScanner

                - bounded : bool
                    Whether or not events must have preceding and 
                    succeeding samples. Default: True
        """
        kwargs = {}
        kwargs["iterables"] = [[None]]
        kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"].copy()

        if "sim_db_file" not in user_kwargs:
            raise ValueError("'sim_db_file' is a required keyword argument.")

        if "scanner_plugin_file" not in user_kwargs:
            raise ValueError(
                "'scanner_plugin_file' is a required keyword argument."
            )

        kwargs["transform"] = self.DEFAULT_RUN_KWARGS["transform"].copy()
        kwargs["transform"].update(
            {
                k: user_kwargs[k]
                for k in self.DEFAULT_RUN_KWARGS["transform"]
                if k in user_kwargs
            }
        )
        kwargs["transform"].update(
            {
                "sim_db_file": user_kwargs.pop("sim_db_file"),
                "scanner_plugin_func": load_plugin_from_path(
                    user_kwargs.pop("scanner_plugin_file")
                )
            }
        )
        super().run(**kwargs)

    def get_load_file_path(self, *args):
        """
        Get the load file path.

        Parameters:
        ----------
        *args:
            Not used, but required to maintain compatibility with base class.
        """
        result = os.path.join(
            self.load_dir,
            "rubin_sim_effective_monitoring_time.parquet"
        )
        return result
