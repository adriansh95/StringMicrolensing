import os
import numpy as np
import pandas as pd
from tests.base_etl_task_test import BaseETLTaskTest
from utils.tasks.sample_good_windows_task import SampleGoodWindowsTask

class TestSampleGoodWindowsTask(BaseETLTaskTest):
    def setUp(self):
        self.iterables = [[0, 1], ["v0"]]
        super().setUp()
        self.summary_table_path = os.path.join(
            self.extract_dir,
            "summary_table.parquet"
        )
        self.get_summary_table().to_parquet(self.summary_table_path)
        self.rng_seed=555
        self.task = SampleGoodWindowsTask(self.extract_dir, self.load_dir)

    def get_extract_file_path(self, i_tau, version):
        result = os.path.join(
            self.extract_dir,
            f"good_windows_tau{i_tau}_{version}.parquet"
        )
        return result

    def get_load_file_path(self, i_tau, version):
        result = os.path.join(
            self.load_dir,
            f"sampled_windows_tau{i_tau}_{version}.parquet"
        )
        return result

    def get_extract_data(self):
        data = {
            "tau_0": np.arange(20)
        }
        idx = pd.MultiIndex.from_arrays(
            [
                ["000"] * 10 + ["001"] * 10,
                np.tile(np.repeat(np.arange(5), 2), 2),
                ["start", "end"] * 10
            ],
            names=["objectid", "number", "boundary"]
        )
        result  = pd.DataFrame(data=data, index=idx)
        return result

    def get_expected_data(self):
        expected_data = {
            "start": [0, 2] * 2,
            "end": [1, 3] * 2,
            "t_start": [0.1, 2.4] * 2
        }
        idx = pd.MultiIndex.from_arrays(
            [
                ["fixed"] * 2 + ["variable"] * 2,
                ["000"] * 4,
                [0, 1] * 2,
            ],
            names=["bandwidth_type", "objectid", "number"]
        )
        result = pd.DataFrame(
            data=expected_data,
            index=idx
        )
        return result

    def get_summary_table(self):
        table_data = {
            "fixed_bw_v0_lc_class": ["unimodal", "background"],
            "variable_bw_v0_lc_class": ["unimodal", "background"],
            "fixed_bw_v1_lc_class": ["unimodal", "background"],
            "variable_bw_v1_lc_class": ["unimodal", "background"],
            "fixed_bw_v2_lc_class": ["unimodal", "background"],
            "variable_bw_v2_lc_class": ["unimodal", "background"]
        }
        result = pd.DataFrame(data=table_data, index=["000", "001"])
        return result

    def test_transform(self):
        transformed_data = self.task.transform(
            self.extract_data,
            0,
            "v0",
            summary_table=self.get_summary_table(),
            n_samples=2,
            rng_seed=self.rng_seed
        )
        expected_data = self.get_expected_data()

        with self.subTest(msg="Check dataframe shape."):
            self.assertEqual(transformed_data.shape, expected_data.shape)

        with self.subTest(msg="Check dataframe index names."):
            self.assertTrue(
                transformed_data.index.name == expected_data.index.name
            )

        with self.subTest(msg="Check dataframe columns."):
            self.assertTrue(
                transformed_data.columns.equals(expected_data.columns)
            )

        with self.subTest(msg="Check dataframe index values."):
            self.assertTrue(
                transformed_data.index.get_level_values(0).equals(
                    expected_data.index.get_level_values(0)
                )
            )

        with self.subTest(msg="Check t_start values."):
            before_end= (
                transformed_data["t_start"] < transformed_data["end"]
            ).all()
            after_start = (
                transformed_data["t_start"] > transformed_data["start"]
            ).all()
            within_bounds = after_start and before_end
            self.assertTrue(within_bounds)

    def test_run(self):
        self.task.run(
            summary_table_path=self.summary_table_path,
            tau_range=self.iterables[0],
            versions=self.iterables[1],
            n_samples=2,
            rng_seed=self.rng_seed
        )

        expected_data = self.get_expected_data()

        for i, f in enumerate(self.test_load_files):
            transformed_data = pd.read_parquet(f)

            with self.subTest(test_number=i, msg="Check dataframe shape."):
                self.assertEqual(transformed_data.shape, expected_data.shape)

            with self.subTest(test_number=i, msg="Check dataframe index names."):
                self.assertTrue(
                    transformed_data.index.name == expected_data.index.name
                )

            with self.subTest(test_number=i, msg="Check dataframe index values."):
                self.assertTrue(
                    transformed_data.index.get_level_values(0).equals(
                        expected_data.index.get_level_values(0)
                    )
                )

            with self.subTest(test_number=i, msg="Check t_start values."):
                before_end= (
                    transformed_data["t_start"] < transformed_data["end"]
                ).all()
                after_start = (
                    transformed_data["t_start"] > transformed_data["start"]
                ).all()
                within_bounds = after_start and before_end
                self.assertTrue(within_bounds)

    def test_get_load_file_path(self):
        for i, f in enumerate(self.test_load_files):
            with self.subTest(test_number=i):
                self.assertEqual(f, self.task.get_load_file_path(i, "v0"))

    def test_get_extract_file_path(self):
        for i, f in enumerate(self.test_extract_files):
            with self.subTest(test_number=i):
                self.assertEqual(f, self.task.get_extract_file_path(i, "v0"))
