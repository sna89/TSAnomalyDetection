from Models.SeasonalESD.seasonal_esd import SeasonalESD
from Models.Arima.arima import Arima
from AnomalyDetectors.esd_ad import ESDAnomalyDetector
from AnomalyDetectors.arima_ad import ArimaAnomalyDetector


class AnomalyDetectionFactory:
    def __init__(self, detector_name, experiment_hyperparameters, model_hyperparameters):
        self.detector_name = detector_name
        self.experiment_hyperparameters = experiment_hyperparameters
        self.model_hyperparameters = model_hyperparameters

    def get_detector(self):
        if self.detector_name == 'esd':
            return ESDAnomalyDetector(SeasonalESD, self.experiment_hyperparameters, self.model_hyperparameters)

        elif self.detector_name == 'arima':
            return ArimaAnomalyDetector(Arima, self.experiment_hyperparameters, self.model_hyperparameters)

        else:
            raise ValueError('No such detector: {}'.format(self.detector_name))