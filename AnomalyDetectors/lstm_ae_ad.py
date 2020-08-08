from AnomalyDetectors.ad import AnomalyDetector
from dataclasses import dataclass


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

    def detect_anomalies(self, df_):
        # model = self.model(df_,
        #                    self.lstm_ae_hyperparameters.seasonality,
        #                    self.experiment_hyperparameters.forecast_period_hours)
        # return model.run()
        pass

