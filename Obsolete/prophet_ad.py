from AnomalyDetectors.ad import AnomalyDetector
from dataclasses import dataclass


@dataclass
class ProphetHyperParameters:
    interval_width: float


class ProphetAnomalyDetector(AnomalyDetector):
    def __init__(self, model, experiment_hyperparameters, model_hyperparameters):
        super(ProphetAnomalyDetector, self).__init__(model, experiment_hyperparameters)
        self.prophet_hyperparameters = ProphetHyperParameters(**model_hyperparameters)

    def detect_anomalies(self, data):
        model = self.model(data,
                           self.prophet_hyperparameters.interval_width,
                           self.experiment_hyperparameters.horizon)

        anomalies = model.run()

        return anomalies
