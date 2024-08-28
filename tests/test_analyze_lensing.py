import unittest
import pandas as pd
import numpy as np
from utils.analyze_lensing import make_lensing_dataframe
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

    def test_make_lensing_dataframe(self):
        input_dataframe = self.input_dataframe
        input_dataframe.loc[4:7, "cluster_label"] = 0
        expected_output_data = {"t_start_max": [3.01],
                                "t_end_max": [8],
                                "t_start_min": [4],
                                "t_end_min": [7.01],
                                "filters": ["rgiz"]}
        idx = pd.MultiIndex.from_tuples([("000", 0)], names=("objectid", None))
        expected_output = pd.DataFrame(data=expected_output_data, index=idx)
        result = make_lensing_dataframe(input_dataframe)
        assert_frame_equal(result, expected_output)

if __name__ == "main":
    unittest.main()