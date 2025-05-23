"""
This module provides the LcScanner and its helper classes.
"""
import numpy as np
from numba import njit

class _FilterState():
    def __init__(self, n_taus):
        self.n_all = np.zeros(7, dtype=int)
        self.n_bright = np.zeros((n_taus, 7), dtype=int)
        self.n_baseline = np.zeros((n_taus, 7), dtype=int)

    def update(self, n_bright):
        """update self"""
        self.n_bright = n_bright
        self.n_baseline = self.n_all - n_bright

    def reset(self):
        """Reset self to initial values"""
        n_taus = self.n_bright.shape[0]
        self.n_all = np.zeros(7, dtype=int)
        self.n_bright = np.zeros((n_taus, 7), dtype=int)
        self.n_baseline = np.zeros((n_taus, 7), dtype=int)

class _TimeState():
    def __init__(self, n_taus):
        self.start_idx = np.zeros(n_taus, dtype=int)
        self.end_idx = np.zeros(n_taus, dtype=int)
        self.t_start = np.zeros(n_taus)

    def update(self, start_idx, end_idx, t_start):
        """update self"""
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.t_start = t_start

    def reset(self):
        """Reset self to initial values"""
        n_taus = self.t_start.shape[0]
        self.start_idx = np.zeros(n_taus, dtype=int)
        self.end_idx = np.zeros(n_taus, dtype=int)
        self.t_start = np.zeros(n_taus)

class _LcScannerState():
    def __init__(self, n_taus):
        self.time = _TimeState(n_taus)
        self.filter = _FilterState(n_taus)
        self.last_window_good = np.full(n_taus, False)
        self.this_window_good = np.full(n_taus, False)

    def reset(self):
        """Reset the scanner's state"""
        self.time.reset()
        self.filter.reset()
        self.last_window_good = np.full(self.this_window_good.shape, False)
        self.this_window_good = np.full(self.this_window_good.shape, False)

class LcScanner():
    """This class includes the record_windows method and its helpers"""
    def __init__(
        self,
        taus,
        n_filters_req=3,
        min_per_filter=2,
        bound_both_sides=True
    ):
        self.taus = taus
        self.n_samples = 0
        self.n_filters_req = n_filters_req
        self.min_per_filter = min_per_filter
        self.state = _LcScannerState(taus.shape[0])
        self.bound_both_sides = bound_both_sides
        self.bounding_indices = (0, -1)

    def update(self, start_idx, end_idx, t_start, n_bright):
        """Update the scanner's state"""
        self.state.time.update(start_idx, end_idx, t_start)
        self.state.filter.update(n_bright)
        self.state.last_window_good = self.state.this_window_good
        self.state.this_window_good = self._good_window()

    def record_windows(self, dataframe):
        """This method is used to scan a lightcurve and find time windows
        during which a lensing event of duration tau could begin and
        would, in principle, be detectable"""
        filter_idx = dataframe["filter_index"].to_numpy()
        sample_times = (
            dataframe["mjd"] + (dataframe["exptime"] / (86400 * 2))
        ).to_numpy()
        result = self._record_windows(sample_times, filter_idx)
        return result

    def _record_windows(self, times, filter_idx):
        self._setup(times, filter_idx)
        result = np.full((self.n_samples * 2, self.taus.shape[0]), np.nan)

        #shift, record, repeat
        for i in range(self.n_samples * 2):
            self._shift_window(times, filter_idx)
            window_changed = (
                self.state.last_window_good != self.state.this_window_good
            )
            result[i] = np.where(
                window_changed,
                self.state.time.t_start,
                np.nan
            )

        d = np.diff(result, axis=0)
        assert((d[np.isfinite(d)] > 0).all())
        result.sort(axis=0)
        result = result[(np.isfinite(result)).any(axis=1)]
        return result

    def _shift_window(self, times, filter_idx):
        dt_start = times[self.state.time.start_idx] - self.state.time.t_start
        end_idx_temp = np.clip(
            self.state.time.end_idx,
            a_min=None,
            a_max=self.n_samples - 1
        )
        dt_end_temp = (
            times[end_idx_temp] - (self.state.time.t_start + self.taus)
        )
        dt_end = np.where(
            self.state.time.end_idx < self.n_samples,
            dt_end_temp,
            np.inf
        )
        smaller_dt_start = dt_start < dt_end
        new_start_idx = np.where(
            smaller_dt_start,
            self.state.time.start_idx + 1,
            self.state.time.start_idx
        )
        new_end_idx = np.where(
            ~smaller_dt_start,
            self.state.time.end_idx + 1,
            self.state.time.end_idx
        )
        dt = np.where(smaller_dt_start, dt_start, dt_end)
        new_t_start = self.state.time.t_start + dt
        new_n_bright = self._compute_n_bright(
            filter_idx,
            new_start_idx,
            new_end_idx
        )
        self.update(new_start_idx, new_end_idx, new_t_start, new_n_bright)

    def _setup(self, times, filter_idx):
        self.state.reset()
        n_samples = len(times)
        self.n_samples = n_samples
        self.state.filter.n_all += np.bincount(filter_idx, minlength=7)

        if self.bound_both_sides:
            self.bounding_indices = (0, n_samples)
        else:
            self.bounding_indices = (-1, n_samples+1)

    def _compute_n_bright(self, filter_idx, start_idx, end_idx):
        result_shape = self.state.filter.n_bright.shape
        result = self._numba_n_bright(
            result_shape,
            filter_idx,
            start_idx,
            end_idx
        )
        return result

    @staticmethod
    @njit
    def _numba_n_bright(shape, filter_idx, start_idx, end_idx):
        result = np.zeros(shape)

        for i in range(shape[0]):
            sti = start_idx[i]
            endi = end_idx[i]
            f_idx = filter_idx[sti:endi]
            result[i] += np.bincount(f_idx, minlength=7)

        return result

    def _good_window(self):
        n_bright = self.state.filter.n_bright
        n_baseline = self.state.filter.n_baseline
        achromatic = self._numba_achromatic(
            n_bright,
            n_baseline,
            self.n_filters_req,
            self.min_per_filter,
            self.taus.shape[0]
        )
        bounded = self._numba_bounded(
            self.state.time.start_idx,
            self.state.time.end_idx,
            self.bounding_indices[0],
            self.bounding_indices[1]
        )
        result = achromatic & bounded
        return result

    @staticmethod
    @njit
    def _numba_bounded(
        start_idx,
        end_idx,
        start_idx_bound,
        end_idx_bound
    ):
        start_bounded = start_idx > start_idx_bound
        end_bounded = end_idx < end_idx_bound
        result = start_bounded & end_bounded
        return result

    @staticmethod
    @njit
    def _numba_achromatic(
        n_bright,
        n_baseline,
        n_filters_req,
        min_per_filter,
        n_taus
    ):
        bright_mask = n_bright > (min_per_filter - 1)
        baseline_mask = n_baseline > 1
        enough_filters = (bright_mask.sum(axis=1) > (n_filters_req - 1))

        # In case I choose not to use Numba here,
        #     bright_and_baseline = (~bright_filter_mask | baseline_filter_mask).all(axis=1)
        # is equivalent but quite a bit slower.
        m = ~bright_mask
        bright_and_baseline = [
            (m[i] | baseline_mask[i]).all() for i in range(n_taus)
        ]
        bright_and_baseline = np.array(bright_and_baseline)
        result = enough_filters & bright_and_baseline
        return result
