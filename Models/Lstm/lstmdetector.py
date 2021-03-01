from Models.anomaly_detection_model import AnomalyDetectionModel
import numpy as np
from Helpers.data_helper import DataConst, DataHelper
import torch
from abc import abstractmethod
from torch.utils.data import TensorDataset, DataLoader
from constants import AnomalyDfColumns
import torch.nn as nn
import pandas as pd
import os


class LstmDetectorConst:
    BOOTSTRAP = 100
    EPOCHS = 1
    N_99_PERCENTILE = 1.66
    EARLY_STOP_EPOCHS = 10


class LstmDetector(AnomalyDetectionModel):
    def __init__(self, model_hyperparameters):
        super(LstmDetector, self).__init__()

        self.model = None
        self.scaler = None
        self.batch_size = None
        self.horizon = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.hidden_dim = model_hyperparameters['hidden_dim']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.horizon = model_hyperparameters['forecast_period']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']
        self.freq = model_hyperparameters['freq']
        self.input_timesteps_period = model_hyperparameters['input_timesteps_period']
        self.categorical_columns = model_hyperparameters['categorical_columns']
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    def fit(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        train_dataset = LstmDetector.get_tensor_dataset(x_train, y_train)
        val_dataset = LstmDetector.get_tensor_dataset(x_val, y_val)

        train_dl = LstmDetector.get_dataloader(train_dataset, self.batch_size)
        val_dl = LstmDetector.get_dataloader(val_dataset, self.batch_size)

        num_features = self.get_num_features(x_train[0])
        self.model = self.get_lstm_model(num_features)
        self.train(train_dl, val_dl)
        return self

    def get_num_features(self, sample):
        total_features = sample.shape[-1]
        num_features = total_features - len(self.categorical_columns)
        return num_features

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        train_df_raw, val_df_raw, test_df_raw = LstmDetector.split_data(data,
                                                                        self.val_ratio,
                                                                        self.input_timesteps_period +
                                                                        self.horizon)

        self.scaler, \
            train_scaled, \
            val_scaled, \
            test_scaled = LstmDetector.scale_data(train_df_raw,
                                                  val_df_raw,
                                                  test_df_raw,
                                                  self.categorical_columns)

        x_train, y_train = self.prepare_data(train_scaled,
                                             self.input_timesteps_period,
                                             self.horizon)

        x_val, y_val = self.prepare_data(val_scaled,
                                         self.input_timesteps_period,
                                         self.horizon)

        x_test, y_test = self.prepare_data(test_scaled,
                                           self.input_timesteps_period,
                                           self.horizon)

        return train_df_raw, val_df_raw, test_df_raw, \
               x_train, y_train, \
               x_val, y_val, \
               x_test, y_test

    def prepare_data(self, data, input_timesteps: int, horizon: int = 0):
        num_samples = data.shape[0] - input_timesteps - horizon + 1
        num_features = self.get_num_features(data[0])

        X = []
        y = []
        for i in range(num_samples):
            X.append(data[i:i + input_timesteps])
            y.append(data[i + input_timesteps: i + input_timesteps + horizon, : num_features])

        return np.array(X), np.array(y)

    @abstractmethod
    def train(self, train_dl, val_dl):
        pass

    @abstractmethod
    def get_lstm_model(self, num_features):
        pass

    def create_anomaly_df(self,
                          seq_mean,
                          seq_lower_bound,
                          seq_upper_bound,
                          seq_labels,
                          index,
                          feature_names):

        seq_len = len(seq_lower_bound)
        num_features = self.get_num_features(seq_lower_bound)

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

    def get_inherent_noise(self, val_dl, h=None, use_hidden=False):
        self.model.eval()

        if use_hidden and h is None:
            h = self.model.init_hidden(self.batch_size)

        loss_function = nn.MSELoss().to(self.device)
        running_val_loss = 0
        for seq, labels in val_dl:
            with torch.no_grad():
                seq = seq.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                # lstm_uncertainty
                if h:
                    y_pred, h = self.model(seq, h)

                # lstm_ae_uncertainty
                else:
                    y_pred = self.model(seq)

                y_pred = y_pred.type(torch.FloatTensor).to(self.device)
                loss = loss_function(y_pred, labels)
                running_val_loss += loss.item()

        running_val_loss /= len(val_dl)
        inherent_noise = running_val_loss
        return inherent_noise

    def predict(self, inputs, bootstrap_iter, inherent_noise, use_hidden):
        self.model.train()

        if use_hidden:
            h = self.model.init_hidden(inputs.size()[0])

            predictions = np.array(
                [self.scaler.inverse_transform(self.model(inputs, h, True)[0].cpu().detach().numpy()) for _ in
                 range(bootstrap_iter)])

        else:
            predictions = np.array(
                [self.scaler.inverse_transform(self.model(inputs)[0].cpu().detach().numpy()) for _ in
                 range(bootstrap_iter)])

        mc_mean = predictions.mean(axis=0)
        mc_var = predictions.var(axis=0)
        uncertainty = np.sqrt(mc_var + inherent_noise)

        lower_bound = mc_mean - LstmDetectorConst.N_99_PERCENTILE * uncertainty
        upper_bound = mc_mean + LstmDetectorConst.N_99_PERCENTILE * uncertainty

        self.model.eval()
        return mc_mean, lower_bound, upper_bound

    @staticmethod
    def identify_anomalies(anomaly_df, num_features):
        for idx in anomaly_df.index:
            idx_df = anomaly_df[anomaly_df.index == idx]
            anomaly_idx_df = idx_df[idx_df[AnomalyDfColumns.IsAnomaly] == 1]
            if not anomaly_idx_df.empty:
                num_anomalies_in_sample = anomaly_idx_df.shape[0]
                if num_anomalies_in_sample >= np.floor(np.sqrt(num_features)):
                    anomaly_df.loc[idx, AnomalyDfColumns.IsAnomaly] = 1
                else:
                    anomaly_df.loc[idx, AnomalyDfColumns.IsAnomaly] = 0
        return anomaly_df

    @staticmethod
    def load_model(model, model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def split_data(data, val_ratio, test_periods):
        train_len = int(data.shape[0] * (1 - val_ratio))
        train_df_raw, val_df_raw = DataHelper.split_train_test(data, train_len)

        val_len = len(val_df_raw) - test_periods
        val_df_raw, test_df_raw = DataHelper.split_train_test(val_df_raw, val_len)

        return train_df_raw, val_df_raw, test_df_raw

    @staticmethod
    def get_tensor_dataset(inputs, labels):
        dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
        return dataset

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, drop_last=True):
        return DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, batch_size=batch_size)

    @staticmethod
    def scale_data(train_df_raw, val_df_raw, test_df_raw, categorical_columns=[]):
        train_df_to_scale, train_df_categorical = DataHelper.split_df_by_columns(train_df_raw, categorical_columns)
        val_df_to_scale, val_df_categorical = DataHelper.split_df_by_columns(val_df_raw, categorical_columns)
        test_df_to_scale, test_df_categorical = DataHelper.split_df_by_columns(test_df_raw, categorical_columns)

        train_scaled, scaler = DataHelper.scale(train_df_to_scale)
        val_scaled = scaler.transform(val_df_to_scale)
        test_scaled = scaler.transform(test_df_to_scale)

        train_scaled = np.append(train_scaled, train_df_categorical.values, axis=1)
        val_scaled = np.append(val_scaled, val_df_categorical.values, axis=1)
        test_scaled = np.append(test_scaled, test_df_categorical.values, axis=1)

        return scaler, train_scaled, val_scaled, test_scaled