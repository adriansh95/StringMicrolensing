import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.base_etl_task_test import BaseETLTaskTest
from utils.tasks.analyze_backgrounds_task import AnalyzeBackgroundsTask

class TestAnalyzeBackgroundsTask(BaseETLTaskTest):
    def setUp(self):
        super().setUp()
        self.task = AnalyzeBackgroundsTask(self.extract_dir, self.load_dir)

    def get_extract_file_path(self, i_batch):
        result = os.path.join(
            self.extract_dir,
            f'kde_labelled_lightcurves_batch{i_batch}.parquet'
        )
        return result

    def get_load_file_path(self, i_batch):
        result = os.path.join(
            self.load_dir,
            f"background_results_batch{i_batch}.parquet"
        )
        return result

    def get_extract_data(self):
        rng = np.random.default_rng(seed=999)
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
        result  = pd.DataFrame(data=data)
        return result

    def get_expected_data(self, i_batch):
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
        result = pd.DataFrame(
            data=expected_data,
            index=pd.MultiIndex.from_arrays(
                [
                    [i_batch, i_batch],
                    ["fixed", "variable"],
                    ["000"] * 2,
                    [0] * 2
                ],
                names=[
                    "batch_number",
                    "bandwidth_type",
                    "objectid",
                    "event_number"
                ]
            )
        )
        return result

    def test_transform(self):
        assert_frame_equal(
            self.get_expected_data(0),
            self.task.transform(self.extract_data, 0)
        )

    def test_run(self, batch_range=(0, 1)):
        self.task.run(batch_range=batch_range)

        for i, f in enumerate(self.test_load_files):
            with self.subTest(test_number=i):
                assert_frame_equal(
                    self.get_expected_data(i),
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
