import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from utils.filtering import unstable_filter

class TestMakeLensingDataframe(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cl = np.ones(20)
        f = np.array(['r', 'g', 'i', 'z'] * 5)
        input_data = {"objectid": np.array(["000"] * 20),
                      "mjd": np.arange(20),
                      "exptime": np.ones(20) * 0.01 * 86400,
                      "cluster_label": cl,
                      "filter": f,
                      "mag_auto": np.ones(20),
                      "magerr_auto": np.ones(20) * 0.05}
        cls.input_dataframe = pd.DataFrame(data=input_data)

    def test_unstable_filter1(self):
        input_dataframe = self.input_dataframe
        expected_output = False
        result = unstable_filter(input_dataframe)
        assert_array_equal(result, expected_output)

if __name__ == "main":
    unittest.main()