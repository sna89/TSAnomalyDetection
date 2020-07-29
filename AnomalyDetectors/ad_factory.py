from SeasonalESD.seasonal_esd import SeasonalESD
from AnomalyDetectors.esd_ad import ESDAnomalyDetector


class AnomalyDetectionFactory:
    def __init__(self, detector, experiment_hyperparameters):
        self.detector = detector
        self.experiment_hyperparameters = experiment_hyperparameters

    def get_detector(self):
        if self.detector == 'esd':
            return ESDAnomalyDetector(SeasonalESD, self.experiment_hyperparameters)
