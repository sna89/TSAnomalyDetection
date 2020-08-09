from Helpers.params_helper import ParamsHelper
from Helpers.data_helper import DataHelper, Period
from datetime import datetime
from Builders.data_builder import PreprocessDataParams
from AnomalyDetectors.ad import ExperimentHyperParameters


DETECTORS = ['esd', 'arima', 'lstm_ae']


class ParamsValidator:
    def __init__(self, params_helper: ParamsHelper):
        self.params_helper = params_helper
        self.detector_name = self.params_helper.get_detector_name()
        self.metadata = self.params_helper.get_metadata()
        self.model_hyperparameters = self.params_helper.get_model_hyperparams()

        self.experiment_hyperparameters = ExperimentHyperParameters(**self.params_helper.get_experiment_hyperparams())
        self.train_period = Period(**self.experiment_hyperparameters.train_period)

        self.preprocess_data_params = PreprocessDataParams(**self.params_helper.get_preprocess_data_params())
        self.test_period = Period(**self.preprocess_data_params.test_period)

    def validate(self):
        self.validate_metadata()

        if self.detector_name in DETECTORS:
            self.validate_uni_variate()
        else:
            raise ValueError('{} detector in not implemented'.format(self.detector_name))

        self.validate_experiment_hyperparameters()
        self.validate_train_time()

    def validate_uni_variate(self):
        num_files = len(self.metadata)
        if num_files > 1:
            raise Exception('{} is uni-variate model. Got {} files in metadata'.format(self.detector_name, num_files))

    def validate_experiment_hyperparameters(self):
        experiment_hyperparameters_keys = list(self.experiment_hyperparameters.__annotations__.keys())

        if 'retrain_schedule_hours' not in experiment_hyperparameters_keys \
            or 'forecast_period_hours' not in experiment_hyperparameters_keys \
                or 'train_period' not in experiment_hyperparameters_keys:
                raise Exception('experiment hyperparamaters need to include: '
                                'retrain_schedule_hours, '
                                'forecast_period_hours, '
                                'train_period')

        retrain_schedule_hours = self.experiment_hyperparameters.retrain_schedule_hours
        forecast_period_hours = self.experiment_hyperparameters.forecast_period_hours

        if retrain_schedule_hours > forecast_period_hours:
            raise ValueError('In experiment hyperparameters:'
                             'retrain_schedule_hours must be lower or equal to forecast_period_hours')

        return

    def validate_metadata(self):
        if not self.metadata:
            raise Exception('No input files')

        for file_metadata in self.metadata:

            file_metadata_keys = list(file_metadata.keys())

            if 'time_column' not in  file_metadata_keys\
                or 'attribute_name' not in file_metadata_keys \
                    or'filename' not in file_metadata_keys:
                        raise Exception(
                            'Missing metadata fields for model {}'.format(self.detector_name))
        return

    def validate_train_time(self):
        is_test = self.preprocess_data_params.test
        if is_test:
            now = datetime.now()
            test_end_time = DataHelper.relative_delta_time(now, hours=self.test_period.hours,
                                                           days=self.test_period.days,
                                                           weeks=self.test_period.weeks)
            train_end_time = DataHelper.relative_delta_time(now, hours=self.train_period.hours,
                                                            days=self.train_period.days,
                                                            weeks=self.train_period.weeks)
            if train_end_time > test_end_time:
                raise ValueError('Initial train epoch time interval must be smaller than test time interval')