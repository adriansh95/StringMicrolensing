import os
import unittest
from itertools import product

class BaseETLTaskTest(unittest.TestCase):
    def setUp(self):
        """
        Set up shared resources or reusable logic for creating test data.
        Subclasses should override `get_input_data` and `get_expected_data`
        to provide task-specific data.
        """
        self.extract_dir = 'tests/test_extract/'
        self.load_dir = 'tests/test_load/'
        iterables = getattr(self, "iterables", [[0, 1]])
        self.iter_product = list(product(*iterables))
        self.test_extract_files = [
            self.get_extract_file_path(*k) for k in self.iter_product
        ]
        self.test_load_files = [
            self.get_load_file_path(*k) for k in self.iter_product
        ]
        self.extract_data = self.get_extract_data()

        for f in self.test_extract_files:
            self.extract_data.to_parquet(f)

    def tearDown(self):
        for f in self.test_extract_files + self.test_load_files:
            if os.path.exists(f):
                os.remove(f)

    def get_extract_file_path(self, *keys):
        """
        Abstract method for getting extract file path.
        Subclasses must override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement get_extract_file_path"
        )

    def get_load_file_path(self, *keys):
        """
        Abstract method for getting load file path.
        Subclasses must override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement get_load_file_path"
        )

    def get_extract_data(self):
        """
        Abstract method for providing extract data.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_extract_data")

    def get_expected_data(self):
        """
        Abstract method for providing expected output data.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_expected_data")
