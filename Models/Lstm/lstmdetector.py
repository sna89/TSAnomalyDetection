from Models.anomaly_detection_model import AnomalyDetectionModel
import numpy as np
from Helpers.data_helper import DataConst
import torch.nn as nn
import torch


class LstmDetector(AnomalyDetectionModel):
    def __init__(self):
        super(LstmDetector, self).__init__()

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