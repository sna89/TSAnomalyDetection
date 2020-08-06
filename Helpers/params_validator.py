from Helpers.params_helper import ParamsHelper


class ParamsValidator:
    def __init__(self, params_helper: ParamsHelper):
        self.params_helper = params_helper
        self.detector_name = self.params_helper.get_detector_name()
        self.metadata = self.params_helper.get_metadata()
        self.model_hyperparameters = self.params_helper.get_model_hyperparams()
        self.experiment_hyperparameters = self.params_helper.get_experiment_hyperparams()

    def validate(self):
        if self.detector_name in ['esd', 'arima']:
            self.validate_uni_variate()
        else:
            raise ValueError('{} detector in not implemented'.format(self.detector_name))

        self.validate_experiment_hyperparameters()
        self.validate_model_hyperparameters()
        self.validate_metadata()

    def validate_uni_variate(self):
        if len(self.metadata) > 1:
            raise Exception('{} is uni-variate model. Got multiple time series in metadata'.format(self.detector_name))

    def validate_model_hyperparameters(self):
        model_hyperparameters_keys = list(self.model_hyperparameters.keys())

        if self.detector_name == 'esd':
            if 'alpha' in model_hyperparameters_keys\
                and 'hybrid' in model_hyperparameters_keys \
                    and 'anomaly_ratio' in model_hyperparameters_keys:
                        return

        if self.detector_name == 'arima':
            if 'seasonality' in model_hyperparameters_keys:
                return

        raise Exception(
            'Missing hyper parameters for model {}'.format(self.detector_name))

    def validate_experiment_hyperparameters(self):
        experiment_hyperparameters_keys = list(self.experiment_hyperparameters.keys())

        if 'retrain_schedule_hours' not in experiment_hyperparameters_keys \
            or 'forecast_period_hours' not in experiment_hyperparameters_keys \
                or 'train_period' not in experiment_hyperparameters_keys:
                raise Exception('experiment hyperparamaters need to include: '
                                'retrain_schedule_hours, '
                                'forecast_period_hours, '
                                'train_period')

        retrain_schedule_hours = self.experiment_hyperparameters['retrain_schedule_hours']
        forecast_period_hours = self.experiment_hyperparameters['forecast_period_hours']

        if retrain_schedule_hours > forecast_period_hours:
            raise ValueError('In experiment hyperparameters:'
                             'retrain_schedule_hours must be lower or equal to forecast_period_hours')

        return

    def validate_metadata(self):
        for file_metadata in self.metadata:

            file_metadata_keys = list(file_metadata.keys())

            if 'time_column' not in  file_metadata_keys\
                or 'attribute_name' not in file_metadata_keys \
                    or'filename' not in file_metadata_keys:
                        raise Exception(
                            'Missing metadata fields for model {}'.format(self.detector_name))
        return