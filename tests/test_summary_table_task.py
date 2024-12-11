import unittest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.tasks.summary_table_task import SummaryTableTask
from utils.helpers import weighted_std

class TestSummaryTableTask(unittest.TestCase):
    def setUp(self):
        # Set up paths for test data and output directories
        self.extract_dir = 'tests/test_extract/'
        self.load_dir = 'tests/test_load/'
        test_extract_file0 = os.path.join(
            self.extract_dir,
            "kde_labelled_lightcurves_batch0.parquet"
            )
        test_extract_file1 = os.path.join(
            self.extract_dir,
            "kde_labelled_lightcurves_batch1.parquet"
            )
        self.test_extract_files = [test_extract_file0, test_extract_file1]
        test_load_file0 = os.path.join(
            self.load_dir,
            "summary_batch0.parquet"
            )
        test_load_file1 = os.path.join(
            self.load_dir,
            "summary_batch1.parquet"
            )
        self.test_load_files = [test_load_file0, test_load_file1]
        config_paths = {"yaml_path": "tests/test_config/test_config.yaml"}
        rng = np.random.default_rng(seed=999)

        # Sample data to write to tests/test_extract
        data = {
            "objectid": ["000"] * 19,
            "filter": ['u', 'g', 'r', 'i'] * 4 + ['u', 'g', 'r'],
            "mjd": list(range(19)),
            "mag_auto": rng.normal(scale=0.001, size=19),
            "magerr_auto": [0.1] * 19,
            "mjd_mid": [1.0005] * 19,
            "bandwidth_variable": [1] * 19,
            "bandwidth_0.13": [1] * 19,
            "exptime": [86.4] * 19
        }
        df = pd.DataFrame(data=data)
        columns=[
            "objectid",
            "mag_auto",
            "magerr_auto",
            "mjd",
            "mjd_mid",
            "exptime",
            "filter",
            "bandwidth_variable",
            "bandwidth_0.13"
        ]
        self.extract_data = df[columns]
        expected = pd.DataFrame()
        sig_cols = ['std_u', 'std_g', 'std_r', 'std_i']
        rms_err_cols = ['rms_err_u', 'rms_err_g', 'rms_err_r', 'rms_err_i']
        lc_class_cols = [
            "fixed_bw_v0_lc_class",
            "fixed_bw_v1_lc_class",
            "fixed_bw_v2_lc_class",
            "variable_bw_v0_lc_class",
            "variable_bw_v1_lc_class",
            "variable_bw_v2_lc_class"
            ]
        sig_vals = [
            weighted_std(
                df.loc[df["filter"] == f, "mag_auto"],
                df.loc[df["filter"] == f, "magerr_auto"]
                )
            for f in ['u', 'g', 'r', 'i']
            ]
        rms_err_vals = [
            np.sqrt(
                np.average(df.loc[df["filter"] == f, "magerr_auto"]**2)
                )
            for f in ['u', 'g', 'r', 'i']
            ]
        lc_class_vals = ["unimodal"] * 6
        expected[sig_cols] = np.array(sig_vals).reshape((1, -1))
        expected[rms_err_cols] = np.array(rms_err_vals).reshape((1, -1))
        expected[lc_class_cols] = np.array(lc_class_vals).reshape((1, -1))
        expected.index = ["000"]
        expected.index.names = ["objectid"]
        self.expected = expected
        self.task = SummaryTableTask(
            self.extract_dir,
            self.load_dir,
            config_paths
            )

        for f in self.test_extract_files:
            df.to_parquet(f)

    def test_transform(self):
        assert_frame_equal(
            self.expected,
            self.task.transform(self.extract_data)
            )

    def test_get_load_file_path(self):
        for i, f in enumerate(self.test_load_files):
            with self.subTest(test_number=i):
                self.assertEqual(f, self.task.get_load_file_path(i))

    def test_get_extract_file_path(self):
        for i, f in enumerate(self.test_extract_files):
            with self.subTest(test_number=i):
                self.assertEqual(f, self.task.get_extract_file_path(i))

    def test_run(self, batch_range=(0, 1)):
        self.task.run(batch_range=batch_range)

        for i, f in enumerate(self.test_load_files):
            with self.subTest(test_number=i):
                assert_frame_equal(
                    self.expected,
                    pd.read_parquet(f)
                    )

    def tearDown(self):
        # Clean up test files
        for f in self.test_extract_files + self.test_load_files:
            if os.path.exists(f):
                os.remove(f)
