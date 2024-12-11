import unittest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.tasks.etl_task import ETLTask

class TestETLTask(unittest.TestCase):
    def setUp(self):
        # Set up paths for test data and output directories
        self.extract_dir = 'tests/test_extract/'
        self.load_dir = 'tests/test_load/'
        self.test_extract_file = os.path.join(
            self.extract_dir,
            "etl_extract.parquet"
            )
        self.test_load_file = os.path.join(
            self.load_dir,
            "etl_load.parquet"
            )

        rng = np.random.default_rng(seed=999)
        # Sample data to write to tests/test_extract
        data = {
            "objectid": ["000"] * 20,
            "filter": ['u', 'g', 'r', 'i'] * 5,
            "mjd": list(range(20)),
            "mag_auto": rng.normal(scale=0.001, size=20),
            "magerr_auto": [0.001] * 20,
            "mjd_mid": [1.0005] * 20,
            "bandwidth_variable": [1] * 20,
            "bandwidth_0.13": [1] * 20,
            "exptime": [86.4] * 20
        }
        df = pd.DataFrame(data=data)
        self.data = df
        self.task = ETLTask(
            self.extract_dir,
            self.load_dir,
            )
        df.to_parquet(self.test_extract_file)

    def test_extract(self):
        extracted_data = self.task.extract(self.test_extract_file)
        assert_frame_equal(extracted_data, self.data)

    def test_load(self):
        self.task.load(self.data, self.test_load_file)
        assert_frame_equal(self.data, pd.read_parquet(self.test_load_file))

    def tearDown(self):
        # Clean up test files
        for f in [self.test_extract_file, self.test_load_file]:
            if os.path.exists(f):
                os.remove(f)
