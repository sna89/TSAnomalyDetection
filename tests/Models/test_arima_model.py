import unittest
from Models.Arima.arima import Arima
import pandas as pd
from pandas.util.testing import assert_series_equal
from Helpers.file_helper import FileHelper
import numpy as np


class TestArimaModel(unittest.TestCase):
    def setUp(self):
        self.arima_params = {'seasonality': 3,
                             'forecast_period_hours': 24}

        df_path = FileHelper.get_file_path('arima_test_synthetic.csv')
        true_anomalies_df_path = FileHelper.get_file_path('test_anomalies.csv')

        self.df = pd.read_csv(df_path, index_col=['sampletime'], parse_dates=['sampletime'])
        self.true_anomalies = pd.read_csv(true_anomalies_df_path, index_col=['Unnamed: 0'], parse_dates=['Unnamed: 0'])

    def test_arima_detect(self):
        arima_model = Arima(self.arima_params)
        arima_model = arima_model.fit(self.df)
        predicted_anomalies = arima_model.detect(self.df)

        expected_anomalies_df_path = FileHelper.get_file_path('arima_predicted_anomalies.csv')
        expected_arima_anomalies = pd.read_csv(expected_anomalies_df_path,
                                                              index_col=['sampletime'],
                                                              parse_dates=['sampletime'],
                                                              squeeze=True)
        expected_arima_anomalies.index = expected_arima_anomalies.index.astype(predicted_anomalies.index.dtype)
        expected_arima_anomalies = expected_arima_anomalies.astype(np.float64)

        assert_series_equal(expected_arima_anomalies, predicted_anomalies)

if __name__ == '__main__':
    unittest.main()