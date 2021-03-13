import unittest
from Obsolete.LstmAE.lstmae import LstmDetectorAE
import pandas as pd
from pandas.testing import assert_frame_equal
from Helpers.file_helper import FileHelper
from Helpers.data_helper import DataHelper


class TestLstmAEModel(unittest.TestCase):
    def setUp(self):
        self.lstm_ae_params = { 'hidden_layer': 128,
                                'dropout': 0.1,
                                'batch_size': 16,
                                'threshold': 99,
                                'forecast_period_hours': 24}

        df_path = FileHelper.get_file_path('test_synthetic.csv')
        true_anomalies_df_path = FileHelper.get_file_path('test_anomalies.csv')

        self.df = pd.read_csv(df_path, index_col=['sampletime'], parse_dates=['sampletime'])
        self.df, self.scaler = DataHelper.scale(self.df, self.lstm_ae_params['forecast_period_hours'])

        self.true_anomalies = pd.read_csv(true_anomalies_df_path, index_col=['Unnamed: 0'], parse_dates=['Unnamed: 0'])

    def test_lstm_ae_detect(self):
        lstm_ae_model = LstmDetectorAE(self.lstm_ae_params)
        lstm_ae_model = lstm_ae_model.fit(self.df)
        predicted_anomalies = self.scaler.inverse_transform(lstm_ae_model.detect(self.df))

        predicted_anomalies.to_csv('lstm_ae_test_anomalies.csv')
        expected_anomalies_df_path = FileHelper.get_file_path('lstm_ae_test_anomalies.csv')
        expected_lstm_ae_anomalies = pd.read_csv(expected_anomalies_df_path,
                                                              index_col=['sampletime'],
                                                              parse_dates=['sampletime'],
                                                              squeeze=True)

        assert_frame_equal(expected_lstm_ae_anomalies, predicted_anomalies)


if __name__ == '__main__':
    unittest.main()