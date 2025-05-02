import unittest
import pandas as pd
import numpy as np
from microlensing.filtering import unstable_filter
from pandas.testing import assert_frame_equal

class TestMakeLensingDataframe(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cl = np.ones(20)
        f = np.array(['r', 'g', 'i', 'z'] * 5)
        input_data = {"objectid": np.array(["000"] * 20),
                      "mjd": np.arange(20),
                      "exptime": np.ones(20) * 0.01 * 86400,
                      "cluster_label": cl,
                      "filter": f}
        cls.input_dataframe = pd.DataFrame(data=input_data)

    def test_unstable_filter1(self):
        input_dataframe = self.input_dataframe
        expected_output = False
        result = unstable_filter(input_dataframe)
        assert_frame_equal(result, expected_output)

if __name__ == "main":
    unittest.main()
