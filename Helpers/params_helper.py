import yaml
from Helpers.file_helper import FileHelper
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union


@dataclass
class Output:
    csv: bool
    plot: bool


@dataclass
class Metadata:
    source: str
    filename: str
    attribute_names : List[str]
    time_column: str


@dataclass
class PreprocessDataParams:
    test: bool
    test_period: Dict
    fill: str
    skiprows: int


@dataclass
class CreateSyntheticData:
    to_create: bool
    num_of_series: int
    filename: str
    higher_freq: bool
    holiday: bool
    weekend: bool


@dataclass
class ExperimentHyperParameters:
    train_period: Dict
    train_freq: Dict
    forecast_period_hours: int
    include_train_time: bool
    remove_outliers: bool
    scale: bool


class ParamsHelper:
    def __init__(self, filename='params.yml'):
        params_path = FileHelper.get_file_path(filename)

        with open(params_path) as file:
            self.params_dict = yaml.load(file, Loader=yaml.FullLoader)

    def get_params(self, param_name):
        try:
            return self.params_dict[param_name]
        except Exception as e:
            raise ValueError("cannot find parameter {}".format(e))

    def get_anomalies(self):
        return self.get_params('anomalies')

    def get_experiment_hyperparams(self):
        return self.get_params('experiment_hyperparameters')

    def get_model_hyperparams(self):
        detector = self.get_detector_name()
        model_hyperparameters_dict = self.get_params('model_hyperparameters')
        model_hyperparameters = model_hyperparameters_dict[detector]
        return model_hyperparameters

    def get_metadata(self):
        return self.get_params('metadata')

    def get_detector_name(self):
        return self.get_params('detector_name')

    def get_preprocess_data_params(self):
        return self.get_params('preprocess_data_params')

    def get_detectors(self):
        model_hyperparameters_dict = self.get_params('model_hyperparameters')
        detectors = list(model_hyperparameters_dict.keys())
        return detectors

    def get_is_output(self):
        return Output(**self.get_params('output'))

    def create_experiment_name(self):
        now = datetime.now()
        date_time = now.strftime("%m%d%Y%H%M%S")
        detector_name = self.get_detector_name()
        model_hyperparameters = self.get_model_hyperparams()
        get_preprocess_data_params = self.get_preprocess_data_params()

        experiment_name = detector_name + '_' + date_time
                          # '_'.join("{}={}".format(key,val) for (key,val) in model_hyperparameters.items()) + '_' + \
                          # '_'.join("{}={}".format(key,val) for (key,val) in get_preprocess_data_params.items() if
                          #          key != 'test_period') +

        return experiment_name

    def get_synthetic_data_params(self):
        return CreateSyntheticData(**self.get_params('create_synthetic_data'))
