import unittest
import numpy as np
from numpy.testing import assert_array_equal
from utils.helpers import get_bounding_idxs, filter_map

class TestHelpers(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cl = [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        f = ['u', 'g', 'r', 'i', 'z', 'Y', "VR"]
        cl = np.array(cl)
        cls.input_cl= cl
        cls.input_filters = np.array(f)

    def test_get_bounding_idxs(self):
        input_data = self.input_cl
        expected_output = np.array([[-1, 1], [1, 5], [7, 10], [11, 13]])
        result = get_bounding_idxs(input_data)
        assert_array_equal(result, expected_output)

    def test_filter_map(self):
        input_data = self.input_filters
        expected_output = np.array([0, 1, 2, 3, 4, 5, 6])
        result = np.array([filter_map(x) for x in input_data])
        assert_array_equal(result, expected_output)

if __name__ == "main":
    unittest.main()