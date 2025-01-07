import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.base_etl_task_test import BaseETLTaskTest
from utils.tasks.summary_table_task import SummaryTableTask
from utils.helpers import weighted_std

class TestSummaryTableTask(BaseETLTaskTest):
    def setUp(self):
        super().setUp()
        config_paths = {"yaml_path": "tests/test_config/test_config.yaml"}
        self.task = SummaryTableTask(
            self.extract_dir,
            self.load_dir,
            config_paths
        )

    def get_extract_file_path(self, i_batch):
        result = os.path.join(
            self.extract_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
        )
        return result

    def get_load_file_path(self, i_batch):
        result = os.path.join(
            self.load_dir,
            f"summary_batch{i_batch}.parquet"
        )
        return result

    def get_extract_data(self):
        rng = np.random.default_rng(seed=999)
        data = {
            "objectid": ["000"] * 19,
            "mag_auto": rng.normal(scale=0.001, size=19),
            "magerr_auto": [0.1] * 19,
            "mjd": list(range(19)),
            "mjd_mid": [1.0005] * 19,
            "exptime": [86.4] * 19,
            "filter": ['u', 'g', 'r', 'i'] * 4 + ['u', 'g', 'r'],
            "bandwidth_variable": [1] * 19,
            "bandwidth_0.13": [1] * 19
        }
        result = pd.DataFrame(data=data)
        return result

    def get_expected_data(self):
        df = self.get_extract_data()
        result = pd.DataFrame()
        sig_cols = ['std_u', 'std_g', 'std_r', 'std_i']
        sig_vals = [
            weighted_std(
                df.loc[df["filter"] == f, "mag_auto"],
                df.loc[df["filter"] == f, "magerr_auto"]
            )
            for f in ['u', 'g', 'r', 'i']
        ]
        result[sig_cols] = np.array(sig_vals).reshape((1, -1))

        rms_err_cols = ['rms_err_u', 'rms_err_g', 'rms_err_r', 'rms_err_i']
        rms_err_vals = [
            np.sqrt(
                np.average(df.loc[df["filter"] == f, "magerr_auto"]**2)
            )
            for f in ['u', 'g', 'r', 'i']
        ]
        result[rms_err_cols] = np.array(rms_err_vals).reshape((1, -1))

        lc_class_cols = [
            "fixed_bw_v0_lc_class",
            "fixed_bw_v1_lc_class",
            "fixed_bw_v2_lc_class",
            "variable_bw_v0_lc_class",
            "variable_bw_v1_lc_class",
            "variable_bw_v2_lc_class"
        ]
        lc_class_vals = ["unimodal"] * 6
        result[lc_class_cols] = np.array(lc_class_vals).reshape((1, -1))
        result.index = ["000"]
        result.index.names = ["objectid"]
        return result

    def test_transform(self):
        assert_frame_equal(
            self.get_expected_data(),
            self.task.transform(self.extract_data)
            )

    def test_run(self, batch_range=(0, 1)):
        self.task.run(batch_range=batch_range)

        for i, f in enumerate(self.test_load_files):
            with self.subTest(test_number=i):
                assert_frame_equal(
                    self.get_expected_data(),
                    pd.read_parquet(f)
                )

    def test_get_load_file_path(self):
        for i, f in enumerate(self.test_load_files):
            with self.subTest(test_number=i):
                self.assertEqual(f, self.task.get_load_file_path(i))

    def test_get_extract_file_path(self):
        for i, f in enumerate(self.test_extract_files):
            with self.subTest(test_number=i):
                self.assertEqual(f, self.task.get_extract_file_path(i))
