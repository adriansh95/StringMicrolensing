import os
import numpy as np
import pandas as pd
import astropy.units as u
from unittest.mock import patch
from pandas.testing import assert_frame_equal
from tests.base_etl_task_test import BaseETLTaskTest
from utils.tasks.event_rate_task import EventRateTask

class TestEventRateTask(BaseETLTaskTest):
    def setUp(self):
        super().setUp()
        self.mock_bins_patch = patch(
            "utils.tasks.event_rate_task.tau_bins", np.arange(11)
        )
        self.mock_event_calculator_patch = patch(
            "utils.tasks.event_rate_task.EventCalculator"
        )
        self.task = EventRateTask(self.extract_dir, self.load_dir)
        self.mock_bins = self.mock_bins_patch.start()
        self.MockEventCalculator = self.mock_event_calculator_patch.start()
        mock_instance = self.MockEventCalculator.return_value
        mock_instance.calculate.side_effect = (
            lambda: setattr(
                mock_instance,
                "results",
                {"eventRates": np.ones(8) / u.day}
            )
        )
        mock_instance.computeLensingTimePDF.side_effect = (
            lambda bins=np.arange(5): (
                (np.ones((8, bins.shape[0] - 1)) / u.day, bins * u.day)
            )
        )

    def tearDown(self):
        super().tearDown()
        self.mock_bins_patch.stop()
        self.mock_event_calculator_patch.stop()

    def get_extract_file_path(self, key):
        result = os.path.join(
            self.extract_dir,
            "binned_objects.parquet"
        )
        return result

    def get_load_file_path(self, source_distance):
        result = os.path.join(
            self.load_dir,
            f"event_rates_{source_distance}.parquet"
        )
        return result

    def get_extract_data(self):
        rng = np.random.default_rng(seed=999)
        data = {
            "count": [2],
            "ra": [79.756579],
            "dec": [-54.724628],
        }
        result  = pd.DataFrame(data=data)
        return result

    def get_expected_data(self):
        expected_data = {
            f"mu_{i}": np.ones(self.mock_bins.shape[0] - 1) * 2
            for i in range(-15, -7) 
        }
        result = pd.DataFrame(
            data=expected_data
        )
        return result

    def test_transform(self):
        assert_frame_equal(
            self.get_expected_data(),
            self.task.transform(self.extract_data, 1)
        )

    def test_run(self, source_distances=[0, 1]):
        self.task.run(source_distances=source_distances)

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

