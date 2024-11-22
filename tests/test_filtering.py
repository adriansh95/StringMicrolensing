import unittest
import numpy as np
import pandas as pd
from utils.filtering import (unstable_filter,
                             lens_filter,
                             unimodal_filter,
                             lightcurve_classifier)

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        stable_lightcurve = {"objectid": np.array(["000"] * 20),
                             "mjd": np.arange(20),
                             "exptime": np.ones(20) * 0.001,
                             "label_column": np.ones(20),
                             "filter": np.array(['r', 'g', 'i', 'z'] * 5),
                             "mag_auto": np.ones(20),
                             "magerr_auto": np.ones(20) * 0.05}
        stable_df = pd.DataFrame(stable_lightcurve)

        unstable_df = stable_df.copy()
        unstable_df.iloc[18:, 3] = -1

        lensed_df = stable_df.copy()
        lensed_df.iloc[2 * np.arange(10), 5] += 0.1
        lensed_df.iloc[:4, 3] = 0
        lensed_df.iloc[:4, 5] += -2.5 * np.log10(2)

        out_of_order_df = lensed_df.sample(frac=1)

        too_bright_df = lensed_df.copy()
        too_bright_df.iloc[0, 5] -= 10

        input_data = {"stable": stable_df,
                      "unstable": unstable_df,
                      "lensed": lensed_df,
                      "out_of_order": out_of_order_df,
                      "too_bright": too_bright_df}
        self.input_data = input_data

class TestUnstableFilter(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.expected = {"stable": False,
                         "unstable": True,
                         "lensed": False,
                         "out_of_order": False,
                         "too_bright": False}

    def test_unstable_filter(self):
        for case_name in self.input_data.keys():
            input_data = self.input_data[case_name]
            expected = self.expected[case_name]

            with self.subTest(case=case_name):
                self.assertEqual(unstable_filter(input_data,
                                                 label_column="label_column"),
                                 expected)

class TestUnimodalFilter(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.expected = {"stable": True,
                         "unstable": False,
                         "lensed": False,
                         "out_of_order": False,
                         "too_bright": False}

    def test_unimodal_filter(self):
        for case_name in self.input_data.keys():
            input_data = self.input_data[case_name]
            expected = self.expected[case_name]

            with self.subTest(case=case_name):
                self.assertEqual(unimodal_filter(input_data,
                                                 label_column="label_column"),
                                 expected)

class TestLensFilter(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.expected = {"stable": [False] * 5,
                         "unstable": [False] * 5,
                         "lensed": [True, True, True, False, False],
                         "out_of_order": [True, True, True, False, False],
                         "too_bright": [False, True, False, False, False]}
        self.params = [
                       {"samples_per_filter": 1,
                        "unique_filters": 2},
                       {"samples_per_filter": 1,
                        "unique_filters": 2,
                        "factor_of_two": False},
                       {"samples_per_filter": 1,
                       "unique_filters": 3},
                       {"samples_per_filter": 1,
                       "unique_filters": 5},
                       {"samples_per_filter": 2,
                       "unique_filters": 2}
                       ]

    def test_lens_filter(self):
        for case_name in self.input_data.keys():
            input_data = self.input_data[case_name]
            expected_results = self.expected[case_name]

            for p, exp in zip(self.params, expected_results):
                with self.subTest(case=case_name, kwargs=p):
                    self.assertEqual(lens_filter(input_data, **p,
                                                 label_column="label_column"),
                                     exp)

class TestLightcurveClassifier(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.expected = {"stable": ["unimodal"] * 5,
                         "unstable": ["unstable"] * 5,
                         "lensed": ["background", "background", "background",
                                    "NA", "NA"],
                         "out_of_order": ["background", "background",
                                          "background", "NA", "NA"],
                         "too_bright": ["NA", "background", "NA", "NA", "NA"]}
        self.params = [
                       {"samples_per_filter": 1,
                        "unique_filters": 2},
                       {"samples_per_filter": 1,
                        "unique_filters": 2,
                        "factor_of_two": False},
                       {"samples_per_filter": 1,
                       "unique_filters": 3},
                       {"samples_per_filter": 1,
                       "unique_filters": 5},
                       {"samples_per_filter": 2,
                       "unique_filters": 2}
                       ]

    def test_lightcurve_classifier(self):
        for case_name in self.input_data.keys():
            input_data = self.input_data[case_name]
            expected_results = self.expected[case_name]

            for p, exp in zip(self.params, expected_results):
                with self.subTest(case=case_name, kwargs=p):
                    result = lightcurve_classifier(input_data, **p,
                                                   label_column="label_column")
                    self.assertEqual(result, exp)



if __name__ == "main":
    unittest.main()
