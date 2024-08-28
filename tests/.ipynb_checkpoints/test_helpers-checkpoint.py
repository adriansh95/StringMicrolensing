import unittest
import numpy as np
from numpy.testing import assert_array_equal
from utils.helpers import get_bounding_idxs

class TestMakeLensingDataframe(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cl = np.ones(20)
        cl[4:8] = 0
        cl[12:18] = 0
        cls.input_data = cl

    def test_make_lensing_dataframe(self):
        input_data = self.input_data
        expected_output = np.array([[3, 8], [11, 18]])
        result = get_bounding_idxs(input_data)
        assert_array_equal(result, expected_output)

if __name__ == "main":
    unittest.main()