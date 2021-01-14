from Models.anomaly_detection_model import AnomalyDetectionModel
import numpy as np
from Helpers.data_helper import DataConst, DataHelper
import torch
from abc import abstractmethod
from torch.utils.data import TensorDataset, DataLoader


class LstmDetectorConst:
    BOOTSTRAP = 100
    EPOCHS = 150
    N_99_PERCENTILE = 2.57
    EARLY_STOP_EPOCHS = 6


class LstmDetector(AnomalyDetectionModel):
    def __init__(self):
        super(LstmDetector, self).__init__()

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

    @abstractmethod
    def predict(self, inputs, bootstrap_iter, scaler):
        pass

    @abstractmethod
    def create_anomaly_df(self,
                          seq_mean,
                          seq_lower_bound,
                          seq_upper_bound,
                          seq_inputs,
                          scaler,
                          index,
                          feature_names):
        pass

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