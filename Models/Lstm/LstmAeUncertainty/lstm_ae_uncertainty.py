from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty_model import LstmAeUncertaintyModel
from Models.Lstm.lstmdetector import LstmDetector, LstmDetectorConst
from Helpers.data_helper import Period
import numpy as np
import pandas as pd
import torch
import os
from constants import AnomalyDfColumns
from Helpers.time_freq_converter import TimeFreqConverter


LSTM_UNCERTAINTY_HYPERPARAMETERS = ['batch_size',
                                    'encoder_dim',
                                    'dropout',
                                    'forecast_period',
                                    'val_ratio',
                                    'lr',
                                    'input_timesteps_period']


class LstmAeUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmAeUncertainty, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
        self.encoder_dim = model_hyperparameters['encoder_dim']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']
        self.freq = model_hyperparameters['freq']

        input_timesteps_period = Period(**model_hyperparameters['input_timesteps_period'])
        self.input_timesteps_period = TimeFreqConverter.convert_to_num_samples(period=input_timesteps_period,
                                                                               freq=self.freq)

        self.horizon = model_hyperparameters['forecast_period']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    @staticmethod
    def prepare_data(data, input_timesteps: int, horizon: int = 0):
        num_samples = data.shape[0] - input_timesteps - horizon + 1

        X = []
        y = []
        for i in range(num_samples):
            X.append(data[i:i + input_timesteps])
            y.append(data[i + input_timesteps: i + input_timesteps + horizon])

        return np.array(X), np.array(y)

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        train_df_raw, val_df_raw, test_df_raw = LstmDetector.split_data(data,
                                                                        self.val_ratio,
                                                                        self.input_timesteps_period + self.horizon)

        self.scaler, \
            train_scaled, \
            val_scaled, \
            test_scaled = LstmDetector.scale_data(train_df_raw, val_df_raw, test_df_raw)

        x_train, y_train, = self.prepare_data(train_scaled, self.input_timesteps_period, self.horizon)
        x_val, y_val = self.prepare_data(val_scaled, self.input_timesteps_period, self.horizon)
        x_test, y_test = self.prepare_data(test_scaled, self.input_timesteps_period, self.horizon)

        return train_df_raw, val_df_raw, test_df_raw, \
               x_train, y_train, \
               x_val, y_val, \
               x_test, y_test

    def fit(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        train_dataset = LstmDetector.get_tensor_dataset(x_train, y_train)
        val_dataset = LstmDetector.get_tensor_dataset(x_val, y_val)

        train_dl = LstmDetector.get_dataloader(train_dataset, self.batch_size)
        val_dl = LstmDetector.get_dataloader(val_dataset, self.batch_size)

        num_features = x_train.shape[2]
        self.model = self.get_lstm_model(num_features)
        self.train(train_dl, val_dl)
        return self

    def get_lstm_model(self, num_features):
        model = LstmAeUncertaintyModel(num_features, self.encoder_dim, self.dropout, self.batch_size, self.horizon, self.device)
        return model.to(self.device)

    def train(self, train_dl, val_dl):
        epochs = LstmDetectorConst.EPOCHS
        early_stop_epochs = LstmDetectorConst.EARLY_STOP_EPOCHS
        lr = self.lr
        model_path = self.model_path

        self.model.train_(train_dl, val_dl, epochs, early_stop_epochs, lr, model_path)

    @validate_anomaly_df_schema
    def detect(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        num_features = x_train.shape[2]

        val_dataset = LstmDetector.get_tensor_dataset(x_val, y_val)
        val_dl = LstmDetector.get_dataloader(val_dataset, self.batch_size)

        test_dataset = LstmDetector.get_tensor_dataset(x_test, y_test)
        inputs, labels = test_dataset.tensors[0], test_dataset.tensors[1]
        inputs = inputs.type(torch.FloatTensor).to(self.device)
        labels = labels.type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        inherent_noise = self.get_inherent_noise(val_dl, use_hidden=False)
        mc_mean, lower_bounds, upper_bounds = self.predict(inputs, LstmDetectorConst.BOOTSTRAP, inherent_noise, False)

        anomaly_df = self.create_anomaly_df(mc_mean,
                                            lower_bounds,
                                            upper_bounds,
                                            labels,
                                            test_df_raw.index,
                                            feature_names=test_df_raw.columns)
        return anomaly_df

    def create_anomaly_df(self,
                          seq_mean,
                          seq_lower_bound,
                          seq_upper_bound,
                          seq_labels,
                          index,
                          feature_names):

        seq_len = len(seq_lower_bound)
        num_features = seq_lower_bound.shape[1]

        dfs = []

        for feature in range(num_features):
            data = {}

            for idx in range(seq_len):
                y = self.scaler.inverse_transform(seq_labels[0][idx].cpu().numpy())[feature]

                sample_mean = seq_mean[idx][feature]
                sample_lower_bound = seq_lower_bound[idx][feature]
                sample_upper_bound = seq_upper_bound[idx][feature]

                index_idx = int(len(index) - self.horizon + idx)
                dt_index = index[index_idx]

                is_anomaly = 1 if (y <= sample_lower_bound) or (y >= sample_upper_bound) else 0

                data[dt_index] = {
                    AnomalyDfColumns.Feature: feature_names[feature],
                    AnomalyDfColumns.Prediction: sample_mean,
                    AnomalyDfColumns.LowerBound: sample_lower_bound,
                    AnomalyDfColumns.UpperBound: sample_upper_bound,
                    AnomalyDfColumns.Actual: y,
                    AnomalyDfColumns.IsAnomaly: is_anomaly
                }

            df = pd.DataFrame.from_dict(data, orient='index')
            dfs.append(df)

        anomaly_df = pd.concat(dfs, axis=0)

        # pd.set_option('display.max_columns', 999)
        # print(anomaly_df)

        anomaly_df = LstmDetector.identify_anomalies(anomaly_df, num_features)
        return anomaly_df






