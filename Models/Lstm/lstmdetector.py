from Models.anomaly_detection_model import AnomalyDetectionModel
import numpy as np
from Helpers.data_helper import DataConst, DataHelper
import torch
from abc import abstractmethod
from torch.utils.data import TensorDataset, DataLoader
from constants import AnomalyDfColumns


class LstmDetectorConst:
    BOOTSTRAP = 100
    EPOCHS = 150
    N_99_PERCENTILE = 2.57
    EARLY_STOP_EPOCHS = 6


class LstmDetector(AnomalyDetectionModel):
    def __init__(self):
        super(LstmDetector, self).__init__()

        self.model = None
        self.scaler = None
        self.batch_size = None

    @abstractmethod
    def init_data(self, data):
        pass

    @staticmethod
    @abstractmethod
    def prepare_data(data, forecast_period_hours: float, horizon_hours: float = 0):
        pass

    @abstractmethod
    def train(self, train_dl, val_dl):
        pass

    @abstractmethod
    def get_lstm_model(self, num_features):
        pass

    def predict(self, inputs, bootstrap_iter, use_hidden):
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
        mc_std = predictions.std(axis=0)
        lower_bound = mc_mean - LstmDetectorConst.N_99_PERCENTILE * mc_std
        upper_bound = mc_mean + LstmDetectorConst.N_99_PERCENTILE * mc_std
        self.model.eval()
        return mc_mean, lower_bound, upper_bound

    @abstractmethod
    def create_anomaly_df(self,
                          seq_mean,
                          seq_lower_bound,
                          seq_upper_bound,
                          seq_inputs,
                          index,
                          feature_names):
        pass

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
    def split_data(data, val_ratio, test_hours):
        val_hours = int(data.shape[0] * val_ratio / DataConst.SAMPLES_PER_HOUR)
        train_df_raw, val_df_raw = DataHelper.split_train_test(data, val_hours)
        val_df_raw, test_df_raw = DataHelper.split_train_test(val_df_raw, test_hours)

        return train_df_raw, val_df_raw, test_df_raw

    @staticmethod
    def get_tensor_dataset(inputs, labels):
        dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
        return dataset

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, drop_last=True):
        return DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, batch_size=batch_size)

    @staticmethod
    def scale_data(train_df_raw, val_df_raw, test_df_raw):
        train_scaled, scaler = DataHelper.scale(train_df_raw)
        val_scaled = scaler.transform(val_df_raw)
        test_scaled = scaler.transform(test_df_raw)

        return scaler, train_scaled, val_scaled, test_scaled