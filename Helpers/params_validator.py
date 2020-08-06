from Helpers.params_helper import ParamsHelper


class ParamsValidator:
    def __init__(self, params_helper: ParamsHelper):
        self.params_helper = params_helper
        self.detector_name = self.params_helper.get_detector_name()
        self.metadata = self.params_helper.get_metadata()
        self.model_hyperparameters = self.params_helper.get_model_hyperparams()

    def validate(self):
        if self.detector_name in ['esd', 'arima']:
            self.validate_uni_variate()
        else:
            raise ValueError('{} detector in not implemented'.format(self.detector_name))
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

    def validate_metadata(self):
        for file_metadata in self.metadata:

            file_metadata_keys = list(file_metadata.keys())

            if 'time_column' not in  file_metadata_keys\
                or 'attribute_name' not in file_metadata_keys \
                    or'filename' not in file_metadata_keys:
                        raise Exception(
                            'Missing metadata fields for model {}'.format(self.detector_name))
        return