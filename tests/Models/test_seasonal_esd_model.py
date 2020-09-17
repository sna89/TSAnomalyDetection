import unittest
from Models.SeasonalESD.seasonal_esd import SeasonalESD
import pandas as pd
from pandas.util.testing import assert_series_equal


class TestEsdModel(unittest.TestCase):
    def setUp(self):
        self.s_esd_params = {'anomaly_ratio': 0.01,
                  'alpha': 0.05,
                  'hybrid': False}

        self.s_h_esd_params = {'anomaly_ratio': 0.01,
                        'alpha': 0.05,
                        'hybrid': True}

        self.df = pd.read_csv('csv/test_synthetic.csv', index_col=['sampletime'], parse_dates=['sampletime'])
        self.true_anomalies = pd.read_csv('csv/test_anomalies.csv', index_col=['Unnamed: 0'], parse_dates=['Unnamed: 0'])

    def test_s_esd_detect(self):
        s_esd = SeasonalESD(**self.s_esd_params)
        s_esd.fit(self.df)
        predicted_anomalies = s_esd.detect(self.df)

        self.expected_s_esd_predicted_anomalies = pd.read_csv('csv/s_esd_predicted_anomalies.csv',
                                                              index_col=['sampletime'],
                                                              parse_dates=['sampletime'],
                                                              squeeze=True)

        assert_series_equal(self.expected_s_esd_predicted_anomalies, predicted_anomalies)

    def test_s_h_esd_detect(self):
        s_h_esd = SeasonalESD(**self.s_h_esd_params)
        s_h_esd.fit(self.df)
        predicted_anomalies = s_h_esd.detect(self.df)

        self.expected_s_h_esd_predicted_anomalies = pd.read_csv('csv/s_h_esd_predicted_anomalies.csv',
                                                                index_col=['sampletime'],
                                                                parse_dates=['sampletime'],
                                                                squeeze=True)

        assert_series_equal(self.expected_s_h_esd_predicted_anomalies, predicted_anomalies)


if __name__ == '__main__':
    unittest.main()