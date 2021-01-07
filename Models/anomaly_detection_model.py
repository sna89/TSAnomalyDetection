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

        err_msg = "Error in anomaly schema validation: {}"
        if isinstance(anomalies, pd.Series):
            anomalies.rename(AnomalyDetectionConst.ATTRIBUTE_NAME, inplace=True)
            assert anomalies.dtype == np.float64, err_msg.format("Anomaly data type should be np.float64")

        elif isinstance(anomalies, pd.DataFrame):
            # num_columns = anomalies.shape[1]
            # assert num_columns == 3, \
            #     'Anomalies dataframe should consist actual value and low and high confidence interval'
            for dtype in anomalies.dtypes:
                assert dtype == np.float64, err_msg.format("Anomaly data type should be np.float64")

        if anomalies.shape[0] > 0:
            assert isinstance(anomalies.index, pd.DatetimeIndex), \
                err_msg.format("Anomaly index data type should be pd.DatetimeIndex")

        return anomalies

    return wrapper


class AnomalyDetectionModel(ABC):
    def __init__(self):
        self.logger = get_logger(__class__.__name__)
        self.anomaly_df = None

    @staticmethod
    def validate_model_hyperpameters(expected_model_hyperparmeters, model_hyperparameters):
        for key in expected_model_hyperparmeters:
            if key not in model_hyperparameters:
                raise "Missing model hyperparameters: {}".format(key)

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a pandas dataframe"
        return

    @staticmethod
    def init_data(data):
        AnomalyDetectionModel._validate_data(data)
        return data

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def detect(self, df):
        pass


