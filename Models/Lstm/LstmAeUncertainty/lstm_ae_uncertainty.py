from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty_model import LstmAeUncertaintyModel
from Models.Lstm.lstmdetector import LstmDetector, LstmDetectorConst
from Helpers.data_helper import DataHelper, DataConst
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os


LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer',
                                    'batch_size',
                                    'encoder_dim',
                                    'dropout',
                                    'forecast_period_hours',
                                    'val_ratio',
                                    'lr']


class LstmAeUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmAeUncertainty, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
        self.hidden_layer = model_hyperparameters['hidden_layer']
        self.encoder_dim = model_hyperparameters['encoder_dim']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']
        self.timesteps_hours = model_hyperparameters['timesteps_hours']

        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']

        self.model = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    @staticmethod
    def prepare_data(data, forecast_period_hours: float, horizon_hours: float = 0):
        forecast_samples = int(forecast_period_hours * DataConst.SAMPLES_PER_HOUR)
        Xs = []
        for i in range(data.shape[0] - forecast_samples + 1):
            Xs.append(data.iloc[i:i + forecast_samples].values)
        return np.array(Xs), np.array(Xs)

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        train_df_raw, val_df_raw, test_df_raw = LstmDetector.split_data(data,
                                                                        self.val_ratio,
                                                                        self.timesteps_hours)

        x_train, y_train = self.prepare_data(train_df_raw, self.timesteps_hours)
        x_val, y_val = self.prepare_data(val_df_raw, self.timesteps_hours)
        x_test, y_test = self.prepare_data(test_df_raw, self.timesteps_hours)

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
        model = LstmAeUncertaintyModel(num_features, self.hidden_layer, self.encoder_dim, self.dropout)
        return model.to(self.device)

    def train(self, train_dl, val_dl):
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf

        early_stop_current_epochs = 0

        for i in range(LstmDetectorConst.EPOCHS):
            running_train_loss = 0
            self.model.train()
            for seq, _ in train_dl:
                optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)

                y_pred = self.model(seq)
                y_pred = y_pred.type(torch.FloatTensor).to(self.device)

                loss = loss_function(y_pred, seq)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            running_train_loss /= len(train_dl)

            running_val_loss = 0
            self.model.eval()

            for seq, _ in val_dl:
                with torch.no_grad():
                    seq = seq.type(torch.FloatTensor).to(self.device)

                    y_pred = self.model(seq)
                    y_pred = y_pred.type(torch.FloatTensor).to(self.device)

                    loss = loss_function(y_pred, seq)

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

        if x_test.shape[0] == 0:
            return pd.DataFrame()

        test_dataset = LstmDetector.get_tensor_dataset(x_test, y_test)
        inputs, _ = test_dataset.tensors[0], test_dataset.tensors[1]
        inputs = inputs.type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        mc_mean, lower_bounds, upper_bounds = self.predict(inputs)

        anomaly_df = self.create_anomaly_df(mc_mean,
                                            lower_bounds,
                                            upper_bounds,
                                            inputs,
                                            None,
                                            test_df_raw.index,
                                            feature_names=test_df_raw.columns)
        return anomaly_df

    def predict(self, inputs, bootstrap_iter=LstmDetectorConst.BOOTSTRAP, scaler=None):
        self.model.train()
        if scaler:
            predictions = np.array(
                [scaler.inverse_transform(self.model(inputs)[0].cpu().detach().numpy()) for _ in
                 range(bootstrap_iter)])
        else:
            predictions = np.array(
                [self.model(inputs)[0].cpu().detach().numpy() for _ in range(bootstrap_iter)])
        mc_mean = predictions.mean(axis=0)
        mc_std = predictions.std(axis=0)
        lower_bound = mc_mean - LstmDetectorConst.N_99_PERCENTILE * mc_std
        upper_bound = mc_mean + LstmDetectorConst.N_99_PERCENTILE * mc_std
        self.model.eval()
        return mc_mean, lower_bound, upper_bound

    def create_anomaly_df(self,
                          seq_mean,
                          seq_lower_bound,
                          seq_upper_bound,
                          seq_inputs,
                          scaler,
                          index,
                          feature_names=['RH', 'Temp']):

        seq_len = len(seq_lower_bound)
        num_features = seq_lower_bound.shape[1]

        dfs = []

        for feature in range(num_features):
            data = {}

            for idx in range(seq_len):
                if scaler:
                    y = scaler.inverse_transform(seq_inputs[idx][0].cpu().numpy())[feature]
                else:
                    y = seq_inputs[0][idx].cpu().numpy()[feature]
                sample_mean = seq_mean[idx][feature]
                sample_lower_bound = seq_lower_bound[idx][feature]
                sample_upper_bound = seq_upper_bound[idx][feature]

                index_idx = idx
                dt_index = index[index_idx]

                is_anomaly = True if (y <= sample_lower_bound) or (y >= sample_upper_bound) else False

                data[dt_index] = {
                    'Feature': feature_names[feature],
                    'mc_mean': sample_mean,
                    'LowerBound': sample_lower_bound,
                    'UpperBound': sample_upper_bound,
                    'y': y,
                    'is_anomaly': is_anomaly
                }

            df = pd.DataFrame.from_dict(data, orient='index')
            dfs.append(df)

        anomaly_df = pd.concat(dfs, axis=0)

        # pd.set_option('display.max_columns', 999)
        # print(anomaly_df)

        for idx in anomaly_df.index:
            idx_df = anomaly_df[anomaly_df.index == idx]
            anomaly_idx_df = idx_df[idx_df['is_anomaly'] == True]
            if not anomaly_idx_df.empty:
                anomaly_df.loc[idx, 'is_anomaly'] = True

        anomaly_df = anomaly_df[anomaly_df['is_anomaly'] == True]
        anomaly_df = anomaly_df.pivot(columns='Feature', values='y')
        return anomaly_df






