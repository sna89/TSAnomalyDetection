import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import functools


class AnomalyDetectionConst:
    ATTRIBUTE_NAME = 'AnomalyValue'


class AnomalyDetectionModel(ABC):
    def __init__(self, data):
        self.data = data

        self.validate_data()
        self.clean_data()

        self.anomaly_df = None
        self.init = False

    def validate_data(self):
        assert isinstance(self.data, pd.DataFrame), "Data must be a pandas dataframe"

    def clean_data(self):
        self.data = self.data.iloc[:, 0]

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def detect(self):
        pass


def validate_anomaly_df_schema(detect):
    @functools.wraps(detect)
    def wrapper(self, *args, **kwargs):
        anomalies = detect(self)

        anomalies.rename(AnomalyDetectionConst.ATTRIBUTE_NAME, inplace=True)

        err_msg = "Error in anomaly schema validation: {}"
        assert isinstance(anomalies, pd.Series), err_msg.format("Anomaly data structure should be Pandas Series")
        assert anomalies.dtype == np.float64, err_msg.format("Anomaly data type should be np.float64")
        assert isinstance(anomalies.index, pd.DatetimeIndex), err_msg.format("Anomaly index data type should be pd.DatetimeIndex")

        return anomalies

    return wrapper