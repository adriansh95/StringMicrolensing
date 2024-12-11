import unittest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.tasks.kde_label_task import KDELabelTask

class TestKDELabelTask(unittest.TestCase):
    def setUp(self):
        # Set up paths for test data and output directories
        self.extract_dir = 'tests/test_extract/'
        self.load_dir = 'tests/test_load/'
        test_extract_file0 = os.path.join(self.extract_dir,
                                       'lightcurves_batch0.parquet')
        test_extract_file1 = os.path.join(self.extract_dir,
                                       'lightcurves_batch1.parquet')
        self.test_extract_files = [test_extract_file0, test_extract_file1]
        test_load_file0 = os.path.join(
            self.load_dir,
            "kde_labelled_lightcurves_batch0.parquet"
            )
        test_load_file1 = os.path.join(
            self.load_dir,
            "kde_labelled_lightcurves_batch1.parquet"
            )
        self.test_load_files = [test_load_file0, test_load_file1]

        rng = np.random.default_rng(seed=999)

        # Sample data to write to tests/test_extract
        data = {
            "objectid": ["000"] * 20,
            "filter": ['u', 'g', 'r', 'i'] * 4 + ['u', 'g', 'r', 'z'],
            "mjd": list(range(20)),
            "mag_auto": rng.normal(scale=0.001, size=20),
            "magerr_auto": [0.1] * 20,
            "mjd_mid": [1.0005] * 20,
            "exptime": [86.4] * 20,
        }
        df = pd.DataFrame(data=data)
        bandwidths = ["variable", 0.13]
        col_names = [f"bandwidth_{bw}" for bw in bandwidths]
        expected = df.copy().iloc[:-1]
        expected[col_names] = 1
        self.extract_data = df
        self.expected = expected
        self.task = KDELabelTask(self.extract_dir, self.load_dir)

        for f in self.test_extract_files:
            df.to_parquet(f)

    def test_transform(self):
        assert_frame_equal(
            self.expected,
            self.task.transform(self.extract_data)
            )

    def test_get_load_file_path(self):
        for i, f in enumerate(self.test_load_files):
            self.assertEqual(f, self.task.get_load_file_path(i))

    def test_get_extract_file_path(self):
        for i, f in enumerate(self.test_extract_files):
            self.assertEqual(f, self.task.get_extract_file_path(i))

    def test_run(self, batch_range=(0, 1)):
        self.task.run(batch_range=batch_range)

        for f in self.test_load_files:
            assert_frame_equal(
                self.expected,
                pd.read_parquet(f)
                )

    def tearDown(self):
        # Clean up test files
        for f in self.test_extract_files + self.test_load_files:
            if os.path.exists(f):
                os.remove(f)
