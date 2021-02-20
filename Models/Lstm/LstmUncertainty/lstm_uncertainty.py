from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmUncertainty.lstm_uncertainty_model import LstmUncertaintyModel
from Models.Lstm.lstmdetector import LstmDetector
from Helpers.data_helper import DataConst, Period
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from Models.Lstm.lstmdetector import LstmDetectorConst
from constants import AnomalyDfColumns
from Helpers.time_freq_converter import TimeFreqConverter

LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer',
                                    'batch_size',
                                    'dropout',
                                    'val_ratio',
                                    'lr',
                                    'input_timesteps_period',
                                    'forecast_period']


class LstmUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmUncertainty, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
        self.hidden_layer = model_hyperparameters['hidden_layer']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.horizon = model_hyperparameters['forecast_period']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']
        self.freq = model_hyperparameters['freq']
        self.input_timesteps_period = model_hyperparameters['input_timesteps_period']
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    @staticmethod
    def prepare_data(data, input_timesteps: float, horizon: int = 0):
        Xs = []
        Ys = []
        for i in range(data.shape[0] - input_timesteps - horizon + 1):
            Xs.append(data[i: i + input_timesteps])
            Ys.append(data[i + input_timesteps: i + input_timesteps + horizon])
        return np.array(Xs), np.array(Ys)

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        train_df_raw, val_df_raw, test_df_raw = LstmDetector.split_data(data,
                                                                        self.val_ratio,
                                                                        self.input_timesteps_period +
                                                                        self.horizon)

        self.scaler, \
            train_scaled, \
            val_scaled, \
            test_scaled = LstmDetector.scale_data(train_df_raw, val_df_raw, test_df_raw)

        x_train, y_train = self.prepare_data(train_scaled, self.input_timesteps_period, self.horizon)
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

    def train(self, train_dl, val_dl):
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf

        early_stop_current_epochs = 0

        for i in range(LstmDetectorConst.EPOCHS):
            h = self.model.init_hidden()

            running_train_loss = 0
            self.model.train()
            for seq, labels in train_dl:
                optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                y_pred, h = self.model(seq, h)
                y_pred = y_pred.type(torch.FloatTensor).to(self.device)

                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            running_train_loss /= len(train_dl)

            running_val_loss = self.get_inherent_noise(val_dl, h, use_hidden=True)

            if i % 10 == 0:
                print(f'epoch: {i:3} train loss: {running_train_loss:10.8f} val loss: {running_val_loss:10.8f}')

            if running_val_loss <= best_val_loss:
                torch.save(self.model.state_dict(), self.model_path)
                best_val_loss = running_val_loss
                early_stop_current_epochs = 0

            else:
                early_stop_current_epochs += 1

            if early_stop_current_epochs == LstmDetectorConst.EARLY_STOP_EPOCHS:
                break

        return

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
        test_inputs, test_labels = test_dataset.tensors[0], test_dataset.tensors[1]
        test_inputs = test_inputs.type(torch.FloatTensor).to(self.device)
        test_labels = test_labels.type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        inherent_noise = self.get_inherent_noise(val_dl, use_hidden=True)
        mc_mean, lower_bounds, upper_bounds = self.predict(test_inputs, LstmDetectorConst.BOOTSTRAP, inherent_noise, True)

        anomaly_df = self.create_anomaly_df(mc_mean[0],
                                            lower_bounds[0],
                                            upper_bounds[0],
                                            test_labels,
                                            test_df_raw.index,
                                            feature_names=test_df_raw.columns)
        return anomaly_df

    def get_lstm_model(self, num_features):
        model = LstmUncertaintyModel(num_features,
                                     self.hidden_layer,
                                     self.batch_size,
                                     self.dropout,
                                     self.horizon,
                                     self.device)
        return model.to(self.device)







