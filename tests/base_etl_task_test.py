import os
import unittest

class BaseETLTaskTest(unittest.TestCase):
    def setUp(self):
        """
        Set up shared resources or reusable logic for creating test data.
        Subclasses should override `get_input_data` and `get_expected_data`
        to provide task-specific data.
        """
        self.extract_dir = 'tests/test_extract/'
        self.load_dir = 'tests/test_load/'
        self.test_extract_files = [
            self.get_extract_file_path(0),
            self.get_extract_file_path(1)
        ]
        self.test_load_files = [
            self.get_load_file_path(0),
            self.get_load_file_path(1)
        ]
        self.extract_data = self.get_extract_data()

        for f in self.test_extract_files:
            self.extract_data.to_parquet(f)

    def tearDown(self):
        for f in self.test_extract_files + self.test_load_files:
            if os.path.exists(f):
                os.remove(f)

    def get_extract_file_path(self, key):
        """
        Abstract method for getting extract file path.
        Subclasses must override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement get_extract_file_path"
        )

    def get_load_file_path(self, key):
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
