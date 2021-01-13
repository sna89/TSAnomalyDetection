# from Models.SeasonalESD.seasonal_esd import SeasonalESD
# from Models.Arima.arima import Arima
from Models.Lstm.LstmAE.lstm_ae import LstmUncertaintyDetectorABCAE
from Models.Lstm.LstmUncertainty.lstm_uncertainty import LstmUncertaintyDetector
# from Models.FBProphet.fbprophet import FBProphet
from AnomalyDetectors.ad import AnomalyDetector


class AnomalyDetectionFactory:
    def __init__(self, detector_name, experiment_hyperparameters, model_hyperparameters):
        self.detector_name = detector_name
        self.experiment_hyperparameters = experiment_hyperparameters
        self.model_hyperparameters = model_hyperparameters

    def get_detector(self):
        # if self.detector_name == 'esd':
        #     return AnomalyDetector(SeasonalESD, self.experiment_hyperparameters, self.model_hyperparameters)

        # elif self.detector_name == 'arima':
        #     return AnomalyDetector(Arima, self.experiment_hyperparameters, self.model_hyperparameters)

        if self.detector_name == 'lstm_ae':
            return AnomalyDetector(LstmUncertaintyDetectorABCAE, self.experiment_hyperparameters, self.model_hyperparameters)

        elif self.detector_name == 'lstm_uncertainty':
            return AnomalyDetector(LstmUncertaintyDetector,
                                   self.experiment_hyperparameters,
                                   self.model_hyperparameters)

        # elif self.detector_name == 'prophet':
        #     return AnomalyDetector(FBProphet, self.experiment_hyperparameters, self.model_hyperparameters)

        else:
            msg = 'No such detector: {}'.format(self.detector_name)
            raise ValueError(msg)