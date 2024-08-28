from collections import defaultdict
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from utils.effective_monitoring_time import effective_monitoring_time

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
        self.input_dataframe = pd.DataFrame(data=input_data)

    def test_effective_monitoring_time(self):
        input_dataframe = pd.DataFrame(data=input_data)
        input_taus = np.array([0.001, 10, 21])
        expected_result = dict()
        expected_result["griz"] = np.array([0.  , 8.91, 0.  ])
        result = dict(effective_monitoring_time(input_dataframe, input_taus))
        assert_array_equal(expected_result["griz"].round(5), result["griz"].round(5))

if __name__ == "main":
    unittest.main()