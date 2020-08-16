from Models.SeasonalESD.seasonal_esd import SeasonalESD
from Models.Arima.arima import Arima
from Models.LstmAE.lstmae import LstmAE
from AnomalyDetectors.esd_ad import ESDAnomalyDetector
from AnomalyDetectors.arima_ad import ArimaAnomalyDetector
from AnomalyDetectors.lstm_ae_ad import LSTMAEAnomalyDetector
from Logger.logger import get_logger


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

        elif self.detector_name == 'lstm_ae':
            return LSTMAEAnomalyDetector(LstmAE, self.experiment_hyperparameters, self.model_hyperparameters)

        else:
            logger = get_logger(__class__.__name__)
            msg = 'No such detector: {}'.format(self.detector_name)
            logger.error(msg)
            raise ValueError(msg)