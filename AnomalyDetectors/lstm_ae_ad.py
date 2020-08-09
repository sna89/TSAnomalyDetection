from AnomalyDetectors.ad import AnomalyDetector
from dataclasses import dataclass
from Helpers.data_helper import DataHelper


@dataclass
class LSTMAEHyperParameters:
    hidden_layer: int
    dropout: float
    batch_size: int
    threshold: float


class LSTMAEAnomalyDetector(AnomalyDetector):
    def __init__(self, model, experiment_hyperparameters, model_hyperparameters):
        super(LSTMAEAnomalyDetector, self).__init__(model, experiment_hyperparameters)
        self.lstm_ae_hyperparameters = LSTMAEHyperParameters(**model_hyperparameters)

    def detect_anomalies(self, data):
        if self.experiment_hyperparameters.scale:
            data, scaler = DataHelper.scale(data, self.experiment_hyperparameters.forecast_period_hours)

        model = self.model(data,
                           self.lstm_ae_hyperparameters.hidden_layer,
                           self.lstm_ae_hyperparameters.dropout,
                           self.lstm_ae_hyperparameters.batch_size,
                           self.lstm_ae_hyperparameters.threshold,
                           self.experiment_hyperparameters.forecast_period_hours)

        anomalies = model.run()
        # TODO inverse scaler on anomalies.
        return anomalies

