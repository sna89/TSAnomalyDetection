import unittest
from Models.SeasonalESD.seasonal_esd import SeasonalESD
import pandas as pd
from pandas.util.testing import assert_series_equal
from Helpers.file_helper import FileHelper

class TestEsdModel(unittest.TestCase):
    def setUp(self):
        self.s_esd_params = {'anomaly_ratio': 0.01,
                  'alpha': 0.05,
                  'hybrid': False}

        self.s_h_esd_params = {'anomaly_ratio': 0.01,
                        'alpha': 0.05,
                        'hybrid': True}

        df_path = FileHelper.get_file_path('test_synthetic.csv')
        true_anomalies_df_path = FileHelper.get_file_path('test_anomalies.csv')

        self.df = pd.read_csv(df_path, index_col=['sampletime'], parse_dates=['sampletime'])
        self.true_anomalies = pd.read_csv(true_anomalies_df_path, index_col=['Unnamed: 0'], parse_dates=['Unnamed: 0'])

    def test_s_esd_detect(self):
        s_esd = SeasonalESD(self.s_esd_params)
        s_esd.fit(self.df)
        predicted_anomalies = s_esd.detect(self.df)

        expected_anomalies_df_path = FileHelper.get_file_path('s_esd_predicted_anomalies.csv')
        self.expected_s_esd_predicted_anomalies = pd.read_csv(expected_anomalies_df_path,
                                                              index_col=['sampletime'],
                                                              parse_dates=['sampletime'],
                                                              squeeze=True)

        assert_series_equal(self.expected_s_esd_predicted_anomalies, predicted_anomalies)

    def test_s_h_esd_detect(self):
        s_h_esd = SeasonalESD(self.s_h_esd_params)
        s_h_esd.fit(self.df)
        predicted_anomalies = s_h_esd.detect(self.df)

        expected_anomalies_df_path = FileHelper.get_file_path('s_h_esd_predicted_anomalies.csv')
        self.expected_s_h_esd_predicted_anomalies = pd.read_csv(expected_anomalies_df_path,
                                                                index_col=['sampletime'],
                                                                parse_dates=['sampletime'],
                                                                squeeze=True)

        assert_series_equal(self.expected_s_h_esd_predicted_anomalies, predicted_anomalies)


if __name__ == '__main__':
    unittest.main()