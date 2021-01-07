from Models.anomaly_detection_model import AnomalyDetectionModel
import numpy as np
from Helpers.data_helper import DataConst


class Lstm(AnomalyDetectionModel):
    def __init__(self):
        super(Lstm, self).__init__()

    @staticmethod
    def prepare_data(data, forecast_period_hours: float, horizon_hours: float = 0):
        forecast_samples = int(forecast_period_hours * DataConst.SAMPLES_PER_HOUR)
        horizon_samples = int(horizon_hours * DataConst.SAMPLES_PER_HOUR)
        Xs = []
        Ys = []
        for i in range(data.shape[0] - forecast_samples - horizon_samples):
            Xs.append(data.iloc[i:i + forecast_samples].values)
            Ys.append(data.iloc[i + forecast_samples : i + forecast_samples + horizon_samples].values)
        return np.array(Xs), np.array(Ys)