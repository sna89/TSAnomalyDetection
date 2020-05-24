from Tasks.task import Task
from dataclasses import dataclass

@dataclass
class SignHyperParameters:
    alpha: float


class SignTask(Task):
    def __init__(self, model, experiment_hyperparameters, attribute='internaltemp'):
        super(SignTask, self).__init__(model, experiment_hyperparameters, attribute)

    def get_model_hyperparameters(self, model_hyperparameters):
        self.model_hyperparameters = SignHyperParameters(**model_hyperparameters)

    def detect_anomalies(self, df_):
        model = self.model(df_, self.model_hyperparameters.alpha)
        return self.run(model)

    def filter_anomalies_in_forecast(self, end_time, detected_anomalies):
        return detected_anomalies