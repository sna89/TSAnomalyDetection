# from Models.SeasonalESD.seasonal_esd import SeasonalESD
# from Models.Arima.arima import Arima
# from Models.Lstm.LstmAE.lstmae import LstmDetectorAE
from Models.Lstm.LstmUncertainty.lstm_uncertainty import LstmUncertainty
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty import LstmAeUncertainty
from Models.Lstm.LstmAeMlpUncertainty.lstm_ae_mlp_uncertainty import LstmAeMlpUncertainty
# from Models.FBProphet.fbprophet import FBProphet
from AnomalyDetectors.ad import AnomalyDetector


class AnomalyDetectionFactory:
    def __init__(self, detector_name, experiment_hyperparameters, model_hyperparameters, freq, categorical_columns=[]):
        self.detector_name = detector_name
        self.experiment_hyperparameters = experiment_hyperparameters
        self.model_hyperparameters = model_hyperparameters
        self.freq = freq
        self.categorical_columns = categorical_columns

    def get_detector(self):
        # if self.detector_name == 'esd':
        #     return AnomalyDetector(SeasonalESD, self.experiment_hyperparameters, self.model_hyperparameters)

        # elif self.detector_name == 'arima':
        #     return AnomalyDetector(Arima, self.experiment_hyperparameters, self.model_hyperparameters)

        # elif self.detector_name == 'prophet':
        #     return AnomalyDetector(FBProphet, self.experiment_hyperparameters, self.model_hyperparameters)

        # if self.detector_name == 'lstm_ae':
        #     return AnomalyDetector(LstmDetectorAE,
        #                            self.experiment_hyperparameters,
        #                            self.model_hyperparameters,
        #                            self.freq)

        if self.detector_name == 'lstm_uncertainty':
            return AnomalyDetector(LstmUncertainty,
                                   self.experiment_hyperparameters,
                                   self.model_hyperparameters,
                                   self.freq,
                                   self.categorical_columns)

        elif self.detector_name == 'lstm_ae_uncertainty':
            return AnomalyDetector(LstmAeUncertainty,
                                   self.experiment_hyperparameters,
                                   self.model_hyperparameters,
                                   self.freq,
                                   self.categorical_columns)

        elif self.detector_name == "lstm_ae_mlp_uncertainty":
            return AnomalyDetector(LstmAeMlpUncertainty,
                                   self.experiment_hyperparameters,
                                   self.model_hyperparameters,
                                   self.freq,
                                   self.categorical_columns)

        else:
            msg = 'No such detector: {}'.format(self.detector_name)
            raise ValueError(msg)