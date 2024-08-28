import unittest
import numpy as np
import pandas as pd
from utils.kde_label import cluster_label_dataframe
from pandas.testing import assert_frame_equal

class TestMakeLensingDataframe(unittest.TestCase):
    @classmethod
    def setUp(cls):
        f = np.array(['r', 'g', 'i', 'z'] * 5)
        input_data = {"objectid": np.array(["000"] * 20),
                      "mjd": np.arange(20),
                      "exptime": np.ones(20) * 0.01 * 86400,
                      "filter": f,
                      "mag_auto": np.ones(20),
                      "magerr_auto": np.ones(20) * 0.05}
        cls.input_dataframe = pd.DataFrame(data=input_data)

    def test_cluster_label_dataframe(self):
        input_dataframe = self.input_dataframe
        input_dataframe.loc[3:6, "mag_auto"] += 3
        expected_output = input_dataframe.copy()
        expected_cl = np.ones(20)
        expected_output["cluster_label"] = expected_cl
        expected_output.loc[3:6, "cluster_label"] = 0
        result = cluster_label_dataframe(input_dataframe)
        assert_frame_equal(result, expected_output)

if __name__ == "main":
    unittest.main()