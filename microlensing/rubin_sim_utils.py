"""
This module provides some utility functions for use with rubin_sim.
Most notably EffectiveMonitoringTimeMetric and
EffectiveMonitoringTimeSummaryMetric.
"""
import inspect
import numpy as np
from rubin_sim import maf
from utils.lc_scanner import LcScanner

def filter_map(char):
    """
    This function assigns a number 0,...,5 to the six Rubin
    filters ugrizy.
    """
    filter_order = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}
    result = filter_order[char]
    return result

vectorized_filter_map = np.vectorize(filter_map)

class EffectiveMonitoringTimeMetric(maf.BaseMetric):
    """
    This metric computes the effective monitoring time at 
    user-defined points on sky.
    """
    def __init__(
        self,
        event_durations,
        filter_col="filter",
        mjd_col="observationStartMJD",
        exp_time_col="visitExposureTime",
        metric_name="EffectiveMonitoringTimeMetric",
        **kwargs
    ):
        self.filter_col = filter_col
        self.mjd_col = mjd_col
        self.exp_time_col = exp_time_col
        self.event_durations = event_durations
        signature = inspect.signature(LcScanner)
        scanner_defaults = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        self.scanner_kwargs = {
            "n_filters_req": kwargs.pop(
                "n_filters_req",
                scanner_defaults["n_filters_req"]
            ),
            "min_per_filter": kwargs.pop(
                "min_per_filter",
                scanner_defaults["min_per_filter"]
            ),
            "bound_both_sides": kwargs.pop(
                "bounded",
                scanner_defaults["bound_both_sides"]
            )
        }
        col = [filter_col, mjd_col, exp_time_col]
        super().__init__(
            col=col,
            units="Days",
            metric_dtype="object",
            metric_name=metric_name,
            **kwargs
    )

    def run(self, data_slice, slice_point=None):
        """Calculate metric values."""
        sort_idxs = np.argsort(data_slice[self.mjd_col])
        sorted_slice = data_slice[sort_idxs]
        f = sorted_slice[self.filter_col]
        filter_index = vectorized_filter_map(f)
        exposure_midpoint = (
            sorted_slice[self.mjd_col] +
            (sorted_slice[self.exp_time_col] / (2 * 86400))
        )
        scanner = LcScanner(self.event_durations, **self.scanner_kwargs)
        windows = scanner._record_windows(exposure_midpoint, filter_index)
        time_delta = windows[1::2] - windows[::2]
        result = np.nansum(time_delta, axis=0) * slice_point["count"]
        return result

class EffectiveMonitoringTimeSummaryMetric(maf.BaseMetric):
    """
    This metric sums the effective monitoring time computed
    by EffectiveMonitoringTimeMetric to give the total EMT.
    """
    def __init__(
        self,
        metric_name="EffectiveMonitoringTimeSummaryMetric",
        **kwargs
    ):
        super().__init__(
            units="Days",
            metric_dtype="object",
            metric_name=metric_name,
            **kwargs
    )

    def run(self, data_slice, slice_point=None):
        """Calculate metric values."""
        result = np.vstack([ds[0] for ds in data_slice]).sum(axis=0)
        return result
