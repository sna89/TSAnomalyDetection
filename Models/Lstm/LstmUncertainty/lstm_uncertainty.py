# from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
# from Models.Lstm.lstm import Lstm
# from Helpers.data_helper import DataHelper, DataConst
# import numpy as np
# import pandas as pd
#
#
# LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer', 'dropout', 'forecast_period_hours', 'val_ratio']
#
#
# class LstmUncertainty(Lstm):
#     def __init__(self, model_hyperparameters):
#         super(LstmUncertainty, self).__init__()
#
#         AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
#         self.hidden_layer = model_hyperparameters['hidden_layer']
#         self.dropout = model_hyperparameters['dropout']
#         self.batch_size = model_hyperparameters['batch_size']
#         self.forecast_period_hours = model_hyperparameters['forecast_period_hours']
#         self.val_ratio = model_hyperparameters['val_ratio']
#
#         self.model = None
#
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     def init_data(self, data):
#         data = AnomalyDetectionModel.init_data(data)
#
#         val_hours = int(data.shape[0] * self.val_ratio / DataConst.SAMPLES_PER_HOUR)
#         train_df_raw, val_df_raw = DataHelper.split_train_test(data, val_hours)
#         val_df_raw, test_df_raw = DataHelper.split_train_test(val_df_raw, int(self.forecast_period_hours * 2))
#
#         x_train, y_train = Lstm.prepare_data(train_df_raw, self.forecast_period_hours)
#         x_val, y_val = Lstm.prepare_data(val_df_raw, self.forecast_period_hours)
#         x_test, y_test = Lstm.prepare_data(test_df_raw, self.forecast_period_hours)
#
#
#         return train_df_raw, val_df_raw, test_df_raw, \
#             x_train, y_train, \
#             x_val, y_val, \
#             x_test, y_test
#
#     def fit(self, data):
#         train_df_raw, val_df_raw, test_df_raw, \
#         x_train, y_train, \
#         x_val, y_val, \
#         x_test, y_test = self.init_data(data)
#
#         timesteps = x_train.shape[1]
#         num_features = x_train.shape[2]
#         self.model = self.get_lstm_model(timesteps, num_features)
#         self.train(x_train, y_train)
#         return self
#
#     @validate_anomaly_df_schema
#     def detect(self, data):
#         num_features = data.shape[1]
#         train_df_raw, val_df_raw, test_df_raw, \
#         x_train, y_train, \
#         x_val, y_val, \
#         x_test, y_test = self.init_data(data)
#
#         if x_test.shape[0] == 0:
#             return pd.DataFrame()
#
#     def get_lstm_model(self, timesteps, num_features):
#         model = None
#         return model
#
#     def train(self, train_data):
#         pass
#
#     def predict(self, data):
#         pred = self.model.predict(data)
#         return pred
#
