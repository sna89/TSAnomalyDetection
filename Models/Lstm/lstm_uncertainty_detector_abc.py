from Models.anomaly_detection_model import AnomalyDetectionModel
import numpy as np
from Helpers.data_helper import DataConst, DataHelper
import torch.nn as nn
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from abc import abstractmethod
import pandas as pd

EPOCHS = 150
EARLY_STOP_EPOCHS = 5
BOOTSTRAP = 100
N_95_PERCENTILE = 2.57


class LstmUncertaintyDetectorABC(AnomalyDetectionModel):
    def __init__(self, model_hyperparameters):
        super(LstmUncertaintyDetectorABC, self).__init__()

        self.hidden_layer = model_hyperparameters['hidden_layer']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.val_ratio = model_hyperparameters['val_ratio']
        self.lr = model_hyperparameters['lr']

        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']
        self.timesteps_hours = model_hyperparameters['timesteps_hours']

        self.model = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.getcwd(), 'lstm_ts.pth')

    @abstractmethod
    def get_lstm_model(self, num_features):
        pass

    @staticmethod
    def get_tensor_dataset(inputs, labels):
        dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
        return dataset

    def get_dataloader(self, dataset, shuffle=False, drop_last=True):
        return DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, batch_size=self.batch_size)

    @staticmethod
    def prepare_data(data, forecast_period_hours: float, horizon_hours: float = 0):
        forecast_samples = int(forecast_period_hours * DataConst.SAMPLES_PER_HOUR)
        horizon_samples = 0 if horizon_hours == 0 else max(1, int(horizon_hours * DataConst.SAMPLES_PER_HOUR))
        Xs = []
        Ys = []
        for i in range(data.shape[0] - forecast_samples - horizon_samples):
            Xs.append(data.iloc[i:i + forecast_samples].values)
            Ys.append(data.iloc[i + forecast_samples : i + forecast_samples + horizon_samples].values)
        return np.array(Xs), np.array(Ys)

    @staticmethod
    def load_model(model, model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def init_data(data, val_ratio, timesteps_hours, forecast_period_hours):
        data = AnomalyDetectionModel.validate_data(data)

        val_hours = int(data.shape[0] * val_ratio / DataConst.SAMPLES_PER_HOUR)
        train_df_raw, val_df_raw = DataHelper.split_train_test(data, val_hours)
        val_df_raw, test_df_raw = DataHelper.split_train_test(val_df_raw, timesteps_hours
                                                              + forecast_period_hours * 2)

        x_train, y_train = LstmUncertaintyDetectorABC.prepare_data(train_df_raw, timesteps_hours, forecast_period_hours)
        x_val, y_val = LstmUncertaintyDetectorABC.prepare_data(val_df_raw, timesteps_hours, forecast_period_hours)
        x_test, y_test = LstmUncertaintyDetectorABC.prepare_data(test_df_raw, timesteps_hours, forecast_period_hours)

        return train_df_raw, val_df_raw, test_df_raw, \
               x_train, y_train, \
               x_val, y_val, \
               x_test, y_test

    def _fit(self, data, use_hidden=True):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = LstmUncertaintyDetectorABC.init_data(data,
                                                              self.val_ratio,
                                                              self.timesteps_hours,
                                                              self.forecast_period_hours)

        train_dataset = LstmUncertaintyDetectorABC.get_tensor_dataset(x_train, y_train)
        val_dataset = LstmUncertaintyDetectorABC.get_tensor_dataset(x_val, y_val)

        train_dl = self.get_dataloader(train_dataset)
        val_dl = self.get_dataloader(val_dataset)

        num_features = x_train.shape[2]
        self.model = self.get_lstm_model(num_features)
        self.train(train_dl, val_dl,epochs=EPOCHS, early_stop_epochs=EARLY_STOP_EPOCHS, use_hidden=use_hidden)
        return self

    def _detect(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        num_features = x_train.shape[2]

        if x_test.shape[0] == 0:
            return pd.DataFrame()

        test_dataset = self.get_tensor_dataset(x_test, y_test)
        inputs, labels = test_dataset.tensors[0], test_dataset.tensors[1]

        inputs = inputs.type(torch.FloatTensor).to(self.device)
        labels = labels.type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmUncertaintyDetectorABC.load_model(self.model, self.model_path)

        mc_mean, lower_bounds, upper_bounds = self.predict(inputs)

        anomaly_df = self.create_anomaly_df(mc_mean,
                                            lower_bounds,
                                            upper_bounds,
                                            labels,
                                            None,
                                            test_df_raw.index,
                                            feature_names=test_df_raw.columns)
        return anomaly_df

    def train(self, train_dl, val_dl, epochs=100, early_stop_epochs=5, use_hidden=True):
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf

        early_stop_current_epochs = 0

        for i in range(epochs):
            if use_hidden:
                h = self.model.init_hidden()

            running_train_loss = 0
            self.model.train()
            for seq, labels in train_dl:
                optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                if use_hidden:
                    y_pred, h = self.model(seq, h)
                else:
                    y_pred = self.model(seq)
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

                    if use_hidden:
                        y_pred, h = self.model(seq, h)
                    else:
                        y_pred = self.model(seq)
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

            if early_stop_current_epochs == early_stop_epochs:
                break

        return

    def predict(self, inputs, bootstrap_iter=BOOTSTRAP, scaler=None):
        self.model.train()
        batch_size = inputs.size()[0]
        h = self.model.init_hidden(batch_size)
        if scaler:
            predictions = np.array(
                [scaler.inverse_transform(self.model(inputs, h, True)[0].cpu().detach().numpy()) for _ in
                 range(bootstrap_iter)])
        else:
            predictions = np.array(
                [self.model(inputs, h, True)[0].cpu().detach().numpy() for _ in range(bootstrap_iter)])
        mc_mean = predictions.mean(axis=0)
        mc_std = predictions.std(axis=0)
        lower_bound = mc_mean - N_95_PERCENTILE * mc_std
        upper_bound = mc_mean + N_95_PERCENTILE * mc_std
        self.model.eval()
        return mc_mean, lower_bound, upper_bound

    def create_anomaly_df(self,
                          batch_mean,
                          batch_lower_bound,
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
                sample_mean = batch_mean[idx][0][feature]
                sample_lower_bound = batch_lower_bound[idx][0][feature]
                sample_upper_bound = batch_upper_bound[idx][0][feature]

                index_idx = int(len(index) - self.forecast_period_hours * DataConst.SAMPLES_PER_HOUR + idx)
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

class LstmModel(nn.Module):
    def __init__(self, num_features, hidden_layer, batch_size, dropout_p, device, batch_first=True):
        super(LstmModel, self).__init__()

        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.n_layers = 2
        self.device = device

        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(self.num_features,
                            self.hidden_layer,
                            dropout=dropout_p,
                            num_layers=self.n_layers,
                            batch_first=batch_first)
        self.linear = nn.Linear(self.hidden_layer, self.num_features)
        self.activation = nn.Tanh()

    def forward(self, input_seq, h, test=None):
        lstm_out, a = self.lstm(input_seq, h)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_layer)
        lstm_out = self.dropout(self.activation(lstm_out))
        predictions = self.linear(lstm_out)
        if not test:
            predictions = predictions.view(self.batch_size, -1, self.num_features)
        else:
            predictions = predictions.view(*input_seq.size())
        predictions = predictions[:, -1:, :]
        return predictions, h

    def init_hidden(self, batch_size=None):
        weight = next(self.parameters()).data

        if not batch_size:
            batch_size = self.batch_size

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_layer).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_layer).zero_().to(self.device))

        return hidden


class LSTM_AE_MODEL(nn.Module):
    def __init__(self, input_dim, hidden_layer, encoder_dim, num_layers, dropout_p=0.2):
        super(LSTM_AE_MODEL, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_dim,
                               self.encoder_dim,
                               num_layers=self.num_layers,
                               dropout=dropout_p,
                               batch_first=True)
        self.decoder = nn.LSTM(self.encoder_dim,
                               self.hidden_layer,
                               dropout=dropout_p,
                               num_layers=self.num_layers,
                               batch_first=True)
        self.fc = nn.Linear(self.hidden_layer, self.input_dim)

        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dropout_2 = nn.Dropout(p=dropout_p)

        self.activation = nn.ReLU()

    def forward(self, enc_input):
        bs = enc_input.size()[0]
        seq_len = enc_input.size()[1]

        enc_out, (hidden_enc, _) = self.encoder(enc_input)
        enc_out = self.dropout_1(self.activation(hidden_enc))
        enc_out = enc_out.view(bs, self.num_layers, self.encoder_dim)  # bs, num_layers, self.encoder_dim

        dec_input = enc_out.repeat(1, seq_len, 1)  # bs, seq_len * num_layers, self.encoder_dim

        dec_out, (hidden_dec, _) = self.decoder(dec_input)
        dec_out = self.dropout_2(self.activation(dec_out))  # bs, seq_len * num_layers, self.hidden_layer

        dec_out = dec_out.contiguous().view(-1, self.hidden_layer)  # bs * seq_len * num_layers, self.hidden_layer

        out = self.fc(dec_out)  # bs * seq_len * num_layers, self.input_dim

        out = out.view(bs, seq_len, self.input_dim)

        return out
