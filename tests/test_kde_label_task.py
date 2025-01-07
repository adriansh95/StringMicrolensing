import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.base_etl_task_test import BaseETLTaskTest
from utils.tasks.kde_label_task import KDELabelTask

class TestKDELabelTask(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.task = KDELabelTask(self.extract_dir, self.load_dir)

    def get_extract_file_path(self, i_batch):
        result = os.path.join(
            self.extract_dir,
            f"kde_labelled_lightcurves_batch{i_batch}.parquet"
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
            "filter": ['u', 'g', 'r', 'i'] * 4 + ['u', 'g', 'r', 'z'],
            "mjd": list(range(20)),
            "mag_auto": rng.normal(scale=0.001, size=20),
            "magerr_auto": [0.1] * 20,
            "mjd_mid": [1.0005] * 20,
            "exptime": [86.4] * 20,
        }
        result = pd.DataFrame(data=data)
        return result

    def get_expected_data(self):
        result = self.get_extract_data().iloc[:-1]
        bandwidths = ["variable", 0.13]
        col_names = [f"bandwidth_{bw}" for bw in bandwidths]
        result[col_names] = 1
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
                    self.get_expected_data(i),
                    pd.read_parquet(f)
                )
