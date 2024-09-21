"""
This module provides the LcScanner and its helper classes.
"""
import numpy as np
from numba import njit

class _FilterState():
    def __init__(self, n_taus):
        self.n_all = np.zeros(7, dtype=np.int32)
        self.n_bright = np.zeros((n_taus, 7), dtype=np.int32)
        self.n_baseline = np.zeros((n_taus, 7), dtype=np.int32)

    def update(self, n_bright):
        """update self"""
        self.n_bright = n_bright
        self.n_baseline = self.n_all - n_bright

class _TimeState():
    def __init__(self, n_taus):
        self.start_idx = np.zeros(n_taus, dtype=np.int32)
        self.end_idx = np.ones(n_taus, dtype=np.int32)

    def update(self, start_idx, end_idx):
        """update self"""
        self.start_idx = start_idx
        self.end_idx = end_idx

class _LcScannerState():
    def __init__(self, n_taus):
        self.time = _TimeState(n_taus)
        self.filter = _FilterState(n_taus)
        self.good_window = np.full(n_taus, False)

class LcScanner():
    """This class includes the record_windows method and its helpers"""
    FILTERS = np.array(['u', 'g', 'r', 'i', 'z', 'Y', "VR"])
    FILTER_ORDER = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5, 'VR': 6}

    def __init__(self, taus, n_filters_req=2):
        n_taus = len(taus)
        self.n_taus = n_taus
        self.taus = taus
        self.n_samples = 0
        self.n_filters_req = n_filters_req
        self.state = _LcScannerState(n_taus)

    def update(self, start_idx, end_idx, n_bright):
        """Update the scanner's state"""
        self.state.time.update(start_idx, end_idx)
        self.state.filter.update(n_bright)
        self.state.good_window = self._is_good_window()

    @classmethod
    def _filter_map(cls, char):
        result = cls.FILTER_ORDER[char]
        return result

    @classmethod
    def filter_map(cls, char):
        """This method maps [u, g, r, i, z, Y] to [0, 1, 2, 3, 4, 5]"""
        mapper = np.vectorize(cls._filter_map)
        result = mapper(char)
        return result

    def record_windows(self, dataframe):
        """This method is used to scan a lightcurve and find time windows
        during which a lensing event of duration tau could begin and
        would, in principle, be detectable"""
        filters = dataframe["filter"].values
        sample_times = (dataframe["mjd"] +
                        (dataframe["exptime"] / (86400 * 2))).values
        result = self._record_windows(sample_times, filters)
        return result

    def _record_windows(self, times, filters):
        self._setup(times, filters)
        result = np.full((self.n_samples * 2, self.n_taus), np.nan)
        result[0] = np.where(self.state.good_window,
                             times[self.state.time.end_idx - 1] - self.taus,
                             np.nan)
        result[1] = np.where(self.state.good_window,
                             times[self.state.time.start_idx] + self.taus,
                             np.nan)

        #shift, record, repeat
        for i in range(self.n_samples - 1): # How many iterations?
            self._shift_window(times, filters)
            result[i] = np.where(self.state.good_window,
                                 self._earliest_start(times),
                                 np.nan)
            result[i+1] = np.where(self.state.good_window,
                                   times[self.state.time.start_idx] + self.taus,
                                   np.nan)

        return result

    def _earliest_start(self, times):
        result = np.maximum(times[self.state.time.start_idx - 1],
                            times[self.state.time.end_idx - 1] - self.taus)
        return result

    def _shift_window(self, times, filters):
        new_start_idx = self.state.time.start_idx + 1
        new_end_idx = self._compute_end_idx(times)
        new_n_bright = self._compute_n_bright(filters, new_start_idx, new_end_idx)
        self.update(new_start_idx, new_end_idx, new_n_bright)

    def _compute_end_idx(self, times):
        old_idx = self.state.time.end_idx
        t_end = times[self.state.time.start_idx] + self.taus
        result = self._numba_end_idx(times,
                                     old_idx,
                                     t_end,
                                     self.n_samples,
                                     self.n_taus)
        return result

    @staticmethod
    @njit
    def _numba_end_idx(times, old_idx, t_end, n_samples, n_taus):
        result = [np.argmax(times[old_idx[i]:] > t_end[i]) + old_idx[i]
                  for i in range(n_taus)]
        result = np.array(result)
        result[result == 0] = n_samples
        return result

    def _setup(self, times, filters):
        n_samples = len(times)
        self.n_samples = n_samples

        # Setup Time stuff
        start_idx = self.state.time.start_idx
        end_idx = self._compute_end_idx(times)

        # Filter stuff
        np.add.at(self.state.filter.n_all,
                  self.filter_map(filters),
                  1)
        n_bright = self._compute_n_bright(filters, start_idx, end_idx)
        self.update(start_idx,
                    end_idx,
                    n_bright)

    def _compute_n_bright(self, filters, start_idx, end_idx):
        result = np.zeros(self.state.filter.n_bright.shape)

        for i in range(self.n_taus):
            endi = end_idx[i]
            f_idx = self.filter_map(filters[start_idx:endi])
            np.add.at(result[i], f_idx, 1)

        return result

    def _is_good_window(self):
        n_bright = self.state.filter.n_bright
        n_baseline = self.state.filter.n_baseline
        result = self._numba_good_window(n_bright,
                                         n_baseline,
                                         self.n_filters_req,
                                         self.n_taus)
        return result

    @staticmethod
    @njit
    def _numba_good_window(n_bright, n_baseline, n_filters_req, n_taus):
        bright_mask = n_bright > 0
        baseline_mask = n_baseline > 0
        enough_filters = (bright_mask.sum(axis=1) > (n_filters_req - 1))

        # In case I choose not to use Numba here,
        #     bright_and_baseline = (~bright_filter_mask | baseline_filter_mask).all(axis=1)
        # is equivalent but quite a bit slower.
        m = ~bright_mask
        bright_and_baseline = [(m[i] | baseline_mask[i]).all()
                               for i in range(n_taus)]
        bright_and_baseline = np.array(bright_and_baseline)
        result = enough_filters & bright_and_baseline
        return result
