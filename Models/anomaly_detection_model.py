import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import functools
from Helpers.data_helper import DataHelper
from Logger.logger import get_logger


class AnomalyDetectionConst:
    ATTRIBUTE_NAME = 'AnomalyValue'


def validate_anomaly_df_schema(detect):
    @functools.wraps(detect)
    def wrapper(self, *args, **kwargs):
        data = args[0]
        anomalies = detect(self, data)

        anomalies.rename(AnomalyDetectionConst.ATTRIBUTE_NAME, inplace=True)

        err_msg = "Error in anomaly schema validation: {}"
        assert isinstance(anomalies, pd.Series), err_msg.format("Anomaly data structure should be Pandas Series")
        assert anomalies.dtype == np.float64, err_msg.format("Anomaly data type should be np.float64")
        if anomalies.shape[0] > 0:
            assert isinstance(anomalies.index, pd.DatetimeIndex), err_msg.format("Anomaly index data type should be pd.DatetimeIndex")

        return anomalies

    return wrapper


class AnomalyDetectionModel(ABC):
    def __init__(self):
        self.logger = get_logger(__class__.__name__)
        self.anomaly_df = None

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a pandas dataframe"
        return

    @staticmethod
    def _clean_data(data):
        data = data.iloc[:, 0]
        return data

    @staticmethod
    def init_data(data):
        AnomalyDetectionModel._validate_data(data)
        data = AnomalyDetectionModel._clean_data(data)
        return data

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def detect(self, df):
        pass

    @staticmethod
    def get_train_set(df, forecast_period_hours):
        train_df, _ = DataHelper.split_train_test(df, forecast_period_hours)
        return train_df

    @staticmethod
    def get_test_set(df, forecast_period_hours):
        _, test_df = DataHelper.split_train_test(df, forecast_period_hours)
        return test_df
