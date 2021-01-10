from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.lstmdetector import LstmDetector, LstmModel
from Helpers.data_helper import DataHelper, DataConst
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import os
from Helpers.file_helper import FileHelper

LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer', 'dropout', 'forecast_period_hours', 'val_ratio', 'lr']
TIMESTEPS_HOURS = 3
BOOTSTRAP = 100
EPOCHS = 150
N_95_PERCENTILE = 2
EARLY_STOP_EPOCHS = 10


class LstmDetectorUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmDetectorUncertainty, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
        self.hidden_layer = model_hyperparameters['hidden_layer']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']

        self.model = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        val_hours = int(data.shape[0] * self.val_ratio / DataConst.SAMPLES_PER_HOUR)
        train_df_raw, val_df_raw = DataHelper.split_train_test(data, val_hours)
        val_df_raw, test_df_raw = DataHelper.split_train_test(val_df_raw, TIMESTEPS_HOURS
                                                              + self.forecast_period_hours * 2)

        x_train, y_train = LstmDetector.prepare_data(train_df_raw, TIMESTEPS_HOURS, self.forecast_period_hours)
        x_val, y_val = LstmDetector.prepare_data(val_df_raw, TIMESTEPS_HOURS, self.forecast_period_hours)
        x_test, y_test = LstmDetector.prepare_data(test_df_raw, TIMESTEPS_HOURS, self.forecast_period_hours)

        return train_df_raw, val_df_raw, test_df_raw, \
               x_train, y_train, \
               x_val, y_val, \
               x_test, y_test

    @staticmethod
    def get_tensor_dataset(inputs, labels):
        dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
        return dataset

    def get_dataloader(self, dataset, shuffle=False, drop_last=True):
        return DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, batch_size=self.batch_size)

    def fit(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        train_dataset = LstmDetectorUncertainty.get_tensor_dataset(x_train, y_train)
        val_dataset = LstmDetectorUncertainty.get_tensor_dataset(x_val, y_val)

        train_dl = self.get_dataloader(train_dataset)
        val_dl = self.get_dataloader(val_dataset)

        num_features = x_train.shape[2]
        self.model = self.get_lstm_model(num_features)
        self.train(train_dl, val_dl)
        return self

    @validate_anomaly_df_schema
    def detect(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        num_features = x_train.shape[2]

        if x_test.shape[0] == 0:
            return pd.DataFrame()

        test_dataset = LstmDetectorUncertainty.get_tensor_dataset(x_test, y_test)
        test_inputs, test_labels = test_dataset.tensors[0], test_dataset.tensors[1]

        test_inputs = test_inputs.type(torch.FloatTensor).to(self.device)
        test_labels = test_labels.type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        test_lower_bounds, test_upper_bounds = self.predict(test_inputs)

        FileHelper.delete_file(self.model_path)

        anomaly_df = LstmDetectorUncertainty.create_anomaly_df(test_lower_bounds,
                                                               test_upper_bounds,
                                                               test_labels,
                                                               None,
                                                               test_df_raw.index,
                                                               feature_names=train_df_raw.columns
                                                               )

        return anomaly_df

    def get_lstm_model(self, num_features):
        model = LstmModel(num_features, self.hidden_layer, self.batch_size, self.dropout, self.device)
        return model.to(self.device)

    def predict(self, test_inputs, bootstrap_iter=BOOTSTRAP, scaler=None):
        self.model.train()
        batch_size = test_inputs.size()[0]
        h = self.model.init_hidden(batch_size)
        if scaler:
            predictions = np.array(
                [scaler.inverse_transform(self.model(test_inputs, h, True)[0].cpu().detach().numpy()) for _ in range(bootstrap_iter)])
        else:
            predictions = np.array([self.model(test_inputs, h, True)[0].cpu().detach().numpy() for _ in range(bootstrap_iter)])
        mc_mean = predictions.mean(axis=0)
        mc_std = predictions.std(axis=0)
        lower_bound = mc_mean - N_95_PERCENTILE * mc_std
        upper_bound = mc_mean + N_95_PERCENTILE * mc_std
        self.model.eval()
        return lower_bound, upper_bound

    @staticmethod
    def create_anomaly_df(batch_lower_bound,
                          batch_upper_bound,
                          batch_labels,
                          scaler,
                          index,
                          feature_names=['RH', 'Temp']):

        bs = len(batch_lower_bound)
        num_features = batch_lower_bound.shape[2]

        dfs = []

        for feature in range(num_features):
            data = {}

            for idx in range(bs):
                if scaler:
                    y = scaler.inverse_transform(batch_labels[idx][0].cpu().numpy())[feature]
                else:
                    y = batch_labels[idx][0].cpu().numpy()[feature]
                sample_lower_bound = batch_lower_bound[idx][0][feature]
                sample_upper_bound = batch_upper_bound[idx][0][feature]
                dt_index = index[idx]
                is_anomaly = True if (y <= sample_lower_bound) or (y >= sample_upper_bound) else False

                data[dt_index] = {
                    'Feature': feature_names[feature],
                    'LowerBound': sample_lower_bound,
                    'UpperBound': sample_upper_bound,
                    'y': y,
                    'is_anomaly': is_anomaly
                }

            df = pd.DataFrame.from_dict(data, orient='index')
            dfs.append(df)

        anomaly_df = pd.concat(dfs, axis=0)

        for idx in anomaly_df.index:
            idx_df = anomaly_df[anomaly_df.index == idx]
            anomaly_idx_df = idx_df[idx_df['is_anomaly'] == True]
            if not anomaly_idx_df.empty:
                anomaly_df.loc[idx, 'is_anomaly'] = True

        anomaly_df = anomaly_df.pivot(columns='Feature', values='y')
        return anomaly_df

    def train(self, train_dl, val_dl):
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf

        early_stop_current_epochs = 0

        for i in range(EPOCHS):
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

            if early_stop_current_epochs == EARLY_STOP_EPOCHS:
                break

        return




