from AnomalyDetectors.ad import AnomalyDetector
from dataclasses import dataclass


@dataclass
class ArimaHyperParameters:
    seasonality: int


class ArimaAnomalyDetector(AnomalyDetector):
    def __init__(self, model, experiment_hyperparameters, model_hyperparameters):
        super(ArimaAnomalyDetector, self).__init__(model, experiment_hyperparameters)
        self.arima_hyperparameters = ArimaHyperParameters(model_hyperparameters)

    def detect_anomalies(self, df_):
        model = self.model(df_, self.arima_hyperparameters.seasonality)
        return model.run()
