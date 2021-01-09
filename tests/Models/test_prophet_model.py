# import unittest
# from Models.FBProphet.fbprophet import FBProphet
# import pandas as pd
# from Helpers.file_helper import FileHelper
#
#
# class TestProphetModel(unittest.TestCase):
#     def setUp(self):
#         self.prophet_params = { 'interval_width': 0.999,
#                                 'changepoint_prior_scale': 0.2,
#                                 'forecast_period_hours': 24}
#
#         df_path = FileHelper.get_file_path('test_synthetic.csv')
#         true_anomalies_df_path = FileHelper.get_file_path('test_anomalies.csv')
#
#         self.df = pd.read_csv(df_path, index_col=['sampletime'], parse_dates=['sampletime'])
#         self.true_anomalies = pd.read_csv(true_anomalies_df_path, index_col=['Unnamed: 0'], parse_dates=['Unnamed: 0'])
#
#     def test_prophet_detect(self):
#         prophet_model = FBProphet(self.prophet_params)
#         prophet_model = prophet_model.fit(self.df)
#         predicted_anomalies = prophet_model.detect(self.df)
#
#
# if __name__ == '__main__':
#     unittest.main()