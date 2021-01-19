from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmUncertainty.lstm_uncertainty_model import LstmUncertaintyModel
from Models.Lstm.lstmdetector import LstmDetector
from Helpers.data_helper import DataHelper, DataConst
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from Models.Lstm.lstmdetector import LstmDetectorConst
from constants import AnomalyDfColumns

LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer',
                                    'batch_size',
                                    'dropout',
                                    'val_ratio',
                                    'lr',
                                    'timesteps_hours',
                                    'forecast_period_hours']


class LstmUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmUncertainty, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
        self.hidden_layer = model_hyperparameters['hidden_layer']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']
        self.timesteps_hours = model_hyperparameters['timesteps_hours']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    @staticmethod
    def prepare_data(data, forecast_period_hours: float, horizon_hours: float = 0):
        forecast_samples = int(forecast_period_hours * DataConst.SAMPLES_PER_HOUR)
        horizon_samples = 0 if horizon_hours == 0 else max(1, int(horizon_hours * DataConst.SAMPLES_PER_HOUR))
        Xs = []
        Ys = []
        for i in range(data.shape[0] - forecast_samples - horizon_samples + 1):
            Xs.append(data[i: i + forecast_samples])
            Ys.append(data[i + forecast_samples: i + forecast_samples + horizon_samples])
        return np.array(Xs), np.array(Ys)

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        train_df_raw, val_df_raw, test_df_raw = LstmDetector.split_data(data,
                                                                        self.val_ratio,
                                                                        self.timesteps_hours +
                                                                        self.forecast_period_hours)

        self.scaler, \
            train_scaled, \
            val_scaled, \
            test_scaled = LstmDetector.scale_data(train_df_raw, val_df_raw, test_df_raw)

        x_train, y_train = self.prepare_data(train_scaled, self.timesteps_hours, self.forecast_period_hours)
        x_val, y_val = self.prepare_data(val_scaled, self.timesteps_hours, self.forecast_period_hours)
        x_test, y_test = self.prepare_data(test_scaled, self.timesteps_hours, self.forecast_period_hours)

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

            running_val_loss = 0
            self.model.eval()

            for seq, labels in val_dl:
                with torch.no_grad():
                    seq = seq.type(torch.FloatTensor).to(self.device)
                    labels = labels.type(torch.FloatTensor).to(self.device)

                    y_pred, h = self.model(seq, h)
                    y_pred = y_pred.type(torch.FloatTensor).to(self.device)

                    loss = loss_function(y_pred, labels)

                    running_val_loss += loss.item()

            running_val_loss /= len(val_dl)
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

        test_dataset = LstmDetector.get_tensor_dataset(x_test, y_test)
        inputs, labels = test_dataset.tensors[0], test_dataset.tensors[1]

        inputs = inputs.type(torch.FloatTensor).to(self.device)
        labels = labels.type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        mc_mean, lower_bounds, upper_bounds = self.predict(inputs, LstmDetectorConst.BOOTSTRAP, True)

        anomaly_df = self.create_anomaly_df(mc_mean,
                                            lower_bounds,
                                            upper_bounds,
                                            labels,
                                            test_df_raw.index,
                                            feature_names=test_df_raw.columns)
        return anomaly_df

    def get_lstm_model(self, num_features):
        model = LstmUncertaintyModel(num_features, self.hidden_layer, self.batch_size, self.dropout, self.device)
        return model.to(self.device)

    def create_anomaly_df(self,
                          batch_mean,
                          batch_lower_bound,
                          batch_upper_bound,
                          batch_labels,
                          index,
                          feature_names):

        bs = len(batch_lower_bound)
        num_features = batch_lower_bound.shape[2]

        dfs = []

        for feature in range(num_features):
            data = {}

            for idx in range(bs):
                y = self.scaler.inverse_transform(batch_labels[idx][0].cpu().numpy())[feature]

                sample_mean = batch_mean[idx][0][feature]
                sample_lower_bound = batch_lower_bound[idx][0][feature]
                sample_upper_bound = batch_upper_bound[idx][0][feature]

                index_idx = int(len(index) - self.forecast_period_hours * DataConst.SAMPLES_PER_HOUR + idx)
                dt_index = index[index_idx]

                is_anomaly = 1 if (y < sample_lower_bound) or (y > sample_upper_bound) else 0

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
        anomaly_df = anomaly_df.astype({AnomalyDfColumns.IsAnomaly: 'int32'})
        # pd.set_option('display.max_columns', 999)
        # print(anomaly_df)

        anomaly_df = LstmDetector.identify_anomalies(anomaly_df, num_features)
        return anomaly_df





