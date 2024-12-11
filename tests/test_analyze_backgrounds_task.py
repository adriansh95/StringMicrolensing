import unittest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.tasks.analyze_backgrounds_task import AnalyzeBackgroundsTask 

class TestAnalyzeBackgroundsTask(unittest.TestCase):
    def setUp(self):
        # Set up paths for test data and output directories
        self.extract_dir = 'tests/test_extract/'
        self.load_dir = 'tests/test_load/'
        test_extract_file0 = os.path.join(
            self.extract_dir,
            'kde_labelled_lightcurves_batch0.parquet'
        )
        test_extract_file1 = os.path.join(
            self.extract_dir,
            'kde_labelled_lightcurves_batch1.parquet'
        )
        self.test_extract_files = [test_extract_file0, test_extract_file1]
        test_load_file0 = os.path.join(
            self.load_dir,
            "background_results_batch0.parquet"
            )
        test_load_file1 = os.path.join(
            self.load_dir,
            "background_results_batch1.parquet"
            )
        self.test_load_files = [test_load_file0, test_load_file1]
        rng = np.random.default_rng(seed=999)

        # Sample data to write to tests/test_extract
        data = {
            "objectid": ["000"] * 20,
            "filter": ['u', 'g', 'r', 'i'] * 5,
            "mjd": list(range(20)),
            "mag_auto": rng.normal(scale=0.001, size=20),
            "magerr_auto": [0.1] * 20,
            "mjd_mid": [1.0005] * 20,
            "exptime": [86.4] * 20,
            "bandwidth_0.13": [1] * 5 + [0] * 8 + [1] * 7,
            "bandwidth_variable": [1] * 5 + [0] * 8 + [1] * 7
        }
        data["mag_auto"][5:13] += -2.5 * np.log10(2)
        self.extract_data = pd.DataFrame(data=data)
        expected_data = {
            "t_start_max": [4.001] * 2,
            "t_end_max": [13] * 2,
            "t_start_min": [5] * 2,
            "t_end_min": [12.001] * 2,
            "n_u": [2] * 2,
            "n_g": [2] * 2,
            "n_r": [2] * 2,
            "n_i": [2] * 2,
            "n_z": [0] * 2,
            "n_Y": [0] * 2,
            "n_samples": [8] * 2
        }
        self.expected = pd.DataFrame(
            data=expected_data,
            index=pd.MultiIndex.from_arrays(
                [
                    ["fixed", "variable"],
                    ["000"] * 2,
                    [0] * 2
                ],
                names=["bandwidth_type", "objectid", "event_number"]
            )
        )
        self.task = AnalyzeBackgroundsTask(self.extract_dir, self.load_dir)

        for f in self.test_extract_files:
            self.extract_data.to_parquet(f)

    def test_transform(self):
        i_batch = 0
        assert_frame_equal(
            pd.concat(
                [self.expected],
                keys=[i_batch],
                names = ["batch_number"]
            ),
            self.task.transform(self.extract_data, 0)
        )

    def test_get_load_file_path(self):
        for i, f in enumerate(self.test_load_files):
            self.assertEqual(f, self.task.get_load_file_path(i))

    def test_get_extract_file_path(self):
        for i, f in enumerate(self.test_extract_files):
            self.assertEqual(f, self.task.get_extract_file_path(i))

    def test_run(self, batch_range=(0, 1)):
        self.task.run(batch_range=batch_range)

        for i_batch, f in enumerate(self.test_load_files):
            assert_frame_equal(
                pd.concat(
                    [self.expected],
                    keys=[i_batch],
                    names = ["batch_number"]
                ),
                pd.read_parquet(f)
            )

    def tearDown(self):
        # Clean up test files
        for f in self.test_extract_files + self.test_load_files:
            if os.path.exists(f):
                os.remove(f)

