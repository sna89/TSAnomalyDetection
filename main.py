from AnomalyDetectors.ad_factory import AnomalyDetectionFactory
import pandas as pd
import sys
import numpy as np
from Logger.logger import get_logger
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataConstructor
from Builders.eval_builder import EvalHelper
from Helpers.data_plotter import DataPlotter
from Helpers.data_creator import DataCreator, DataCreatorMetadata
from Helpers.data_helper import DataHelper, Period
import warnings
from Helpers.file_helper import FileHelper
import os
from constants import Paths

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)


def get_and_validate_parameters(filename='params.yml'):
    params_helper = ParamsHelper(filename)
    ParamsValidator(params_helper).validate()
    return params_helper


def create_synthetic_data(synthetic_data_params, output_path):
    filename = synthetic_data_params.filename
    num_of_series = synthetic_data_params.num_of_series
    higher_freq = synthetic_data_params.higher_freq
    weekend = synthetic_data_params.weekend
    holiday = synthetic_data_params.holiday
    period = Period(**synthetic_data_params.period)
    freq = synthetic_data_params.freq

    data_creator = DataCreator()

    df, anomalies_df = data_creator.create_dataset(period,
                                                   freq,
                                                   higher_freq,
                                                   weekend,
                                                   holiday,
                                                   num_of_series)
    data_path = os.path.join(output_path, filename)
    data_creator.save_to_csv(df, data_path)
    return df, anomalies_df


def contruct_data(params_helper):
    metadata = params_helper.get_metadata()
    preprocess_data_params = params_helper.get_preprocess_data_params()
    data = DataConstructor(metadata, preprocess_data_params).read().build()
    return data


def get_data(params_helper):
    synthetic_data_params = params_helper.get_synthetic_data_params()

    anomalies_file_name = params_helper.get_anomalies_file_name()
    anomalies_file_path = os.path.join(Paths.output_path, anomalies_file_name) if anomalies_file_name else None

    anomalies_true_df = pd.DataFrame()
    if synthetic_data_params.to_create:
        _, anomalies_true_df = create_synthetic_data(synthetic_data_params, Paths.output_path)
        if anomalies_file_path:
            anomalies_true_df.to_csv(anomalies_file_path)

    if anomalies_file_path:
        anomalies_index = params_helper.get_anomalies_index()
        anomalies_true_df = pd.read_csv(anomalies_file_path,
                                        index_col=[anomalies_index])
        anomalies_true_df.index = pd.to_datetime(anomalies_true_df.index)

    data = contruct_data(params_helper)
    return data, anomalies_true_df


def run_experiment(params_helper, data):
    detector_name = params_helper.get_detector_name()
    experiment_hyperparameters = params_helper.get_experiment_hyperparams()
    model_hyperparameters = params_helper.get_model_hyperparams()
    detector = AnomalyDetectionFactory(detector_name,
                                       experiment_hyperparameters,
                                       model_hyperparameters).get_detector()

    logger = get_logger('run_experiment')
    logger.info("Starting experiment for anomaly detector: {}".format(detector_name))
    anomalies = detector.run_anomaly_detection_experiment(data)
    return anomalies


def output_results(params_helper, data, anomalies_pred_df, anomalies_true_df=pd.DataFrame()):
    output = params_helper.get_output()
    experiment_name = params_helper.create_experiment_name()

    if output.csv:
        anomalies_pred_df.to_csv(os.path.join(Paths.output_path, '{}.csv'.format(experiment_name)))

    if output.plot:
        features_to_plot = output.features_to_plot
        DataPlotter.plot_anomalies(data=data,
                                   predicted_anomaly_df=anomalies_pred_df,
                                   actual_anomaly_df=anomalies_true_df,
                                   features_to_plot=features_to_plot)

    evaluate_experiment(data, anomalies_pred_df, anomalies_true_df, params_helper)


def evaluate_experiment(data, anomalies_pred_df, anomalies_true_df=pd.DataFrame(), params_helper={}):
    if anomalies_true_df.empty:
        return
    else:
        evaluator = EvalHelper(data, anomalies_true_df, anomalies_pred_df)
        evaluator.build()
        evaluator.output_confusion_matrix()
        evaluator.output_classification_report()
        evaluator.output_auc()
        evaluator.calc_coverage()
        evaluator.output_params(params_helper)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    logger = get_logger('Main')
    logger.info('Starting')

    FileHelper.create_directory(Paths.output_path)
    try:
        params_helper = get_and_validate_parameters()

        data, anomalies_true_df = get_data(params_helper)
        DataPlotter.plot_ts_data(data)

        anomalies_pred_df = run_experiment(params_helper, data)
        output_results(params_helper, data, anomalies_pred_df, anomalies_true_df)

    except Exception as e:
        logger.error(e)

    finally:
        logger.info('Finished')
