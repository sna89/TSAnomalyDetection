from Helpers.params_helper import ParamsHelper, Metadata
from Helpers.data_helper import DataHelper, Period, DataConst
from datetime import datetime
from Builders.data_builder import PreprocessDataParams
from AnomalyDetectors.ad import ExperimentHyperParameters
from Logger.logger import get_logger


class ParamsValidator:
    def __init__(self, params_helper: ParamsHelper):
        self.params_helper = params_helper

        self.detector_name = self.params_helper.get_detector_name()
        self.metadata = self.params_helper.get_metadata()
        self.model_hyperparameters = self.params_helper.get_model_hyperparams()
        self.detectors = self.params_helper.get_detectors()

        self.experiment_hyperparameters = ExperimentHyperParameters(**self.params_helper.get_experiment_hyperparams())
        self.train_period = Period(**self.experiment_hyperparameters.train_period)
        self.train_freq = Period(**self.experiment_hyperparameters.train_freq)

        self.preprocess_data_params = PreprocessDataParams(**self.params_helper.get_preprocess_data_params())
        self.test_period = Period(**self.preprocess_data_params.test_period)

        self.synthetic_data_params = self.params_helper.get_synthetic_data_params()

        self.logger = get_logger(__class__.__name__)

    def validate(self):
        self.logger.info("Validating experiment parameters")

        self.validate_metadata()

        if self.detector_name in self.detectors:
            self.validate_uni_variate()
        else:
            msg = '{} detector in not implemented'.format(self.detector_name)
            raise ValueError(msg)

        self.validate_experiment_hyperparameters()
        self.validate_train_time()
        self.validate_preprocess_data_params()
        self.validate_data_creator()
        self.validate_datectors()

        self.logger.info("Experiment parameters validated successfully")

    def validate_datectors(self):
        base_msg = '{model} model must run with {recommendation}'

        if self.detector_name == 'arima':
            if self.preprocess_data_params.skiprows <= 0:
                msg = base_msg.format(model=self.detector_name,
                                      recommendation="skiprows greater than 0. recommended value = 6")
                raise ValueError(msg)

        if self.detector_name == 'lstm_ae':
            if not self.experiment_hyperparameters.scale:
                msg = base_msg.format(model=self.detector_name,
                                      recommendation="scale applied to data.")
                raise ValueError(msg)

        if self.detector_name == 'prophet':
            allowed_fill_methods = DataConst.FILL_METHODS.copy()
            allowed_fill_methods.remove('ignore')

            # if self.preprocess_data_params.fill == 'ignore':
            #     msg = base_msg.format(model=self.detector_name,
            #                           recommendation="no missing data points. "
            #                                          "preprocess_data_params::fill must get one of the "
            #                                          "following options: {}"
            #                                          .format(allowed_fill_methods))
            #     raise ValueError(msg)

    def validate_uni_variate(self):
        num_files = len(self.metadata)
        if num_files > 1:
            msg = '{} is uni-variate model. Got {} files in metadata'.format(self.detector_name, num_files)
            raise Exception(msg)

        num_attributes = len(self.metadata[0]['attribute_names'])
        if num_attributes > 1:
            msg = '{} is uni-variate model. Got {} attributes in metadata'.format(self.detector_name, num_attributes)
            raise Exception(msg)

    def validate_experiment_hyperparameters(self):
        experiment_hyperparameters_keys = list(self.experiment_hyperparameters.__annotations__.keys())

        if 'forecast_period_hours' not in experiment_hyperparameters_keys \
                or 'train_period' not in experiment_hyperparameters_keys \
                or 'train_freq' not in experiment_hyperparameters_keys:
            msg = 'experiment hyperparamaters need to include: ' \
                  'retrain_schedule_hours, ' \
                  'forecast_period_hours, ' \
                  'train_period'
            raise Exception(msg)

        return

    def validate_metadata(self):
        if not self.metadata:
            raise Exception('No input files')

        for file_metadata in self.metadata:

            file_metadata_keys = list(file_metadata.keys())

            if 'time_column' not in file_metadata_keys \
                    or 'attribute_names' not in file_metadata_keys \
                    or 'filename' not in file_metadata_keys:
                msg = 'Missing metadata fields for model {}'.format(self.detector_name)
                raise Exception(msg)
        return

    def get_filenames(self):
        return [Metadata(**file_metadata).filename for file_metadata in self.metadata]

    def validate_train_time(self):
        is_test = self.preprocess_data_params.test
        if is_test:
            now = datetime.now()
            test_end_time = DataHelper.relative_delta_time(now,
                                                           minutes=0,
                                                           hours=self.test_period.hours,
                                                           days=self.test_period.days,
                                                           weeks=self.test_period.weeks)
            train_end_time = DataHelper.relative_delta_time(now,
                                                            minutes=0,
                                                            hours=self.train_period.hours,
                                                            days=self.train_period.days,
                                                            weeks=self.train_period.weeks)
            if train_end_time > test_end_time:
                msg = 'Initial train epoch time interval must be smaller than test time interval'
                raise ValueError(msg)

    def validate_preprocess_data_params(self):
        if self.preprocess_data_params.fill not in DataConst.FILL_METHODS \
                and self.preprocess_data_params.fill is not None:
            msg = 'pre-process fill method is not supported or missing.'
            raise ValueError(msg)

        if self.preprocess_data_params.skiprows < 0:
            msg = 'skiprows parameter must be greated than 0.'
            raise ValueError(msg)

    def validate_data_creator(self):
        if self.synthetic_data_params.to_create:
            filenames = self.get_filenames()
            if self.synthetic_data_params.filename not in filenames:
                msg = 'new created data not in metadata'
                raise ValueError(msg)

            num_of_series = self.synthetic_data_params.num_of_series
            if num_of_series <= 0:
                msg = 'Number of time series in synthetic dataset must be greater than 0'
                raise ValueError(msg)

            higher_freq = self.synthetic_data_params.higher_freq
            if num_of_series > 1 and higher_freq:
                msg = 'higher freq is implemented only for single time series in synthetic dataset'
                raise ValueError(msg)
