from AnomalyDetectors.ad_factory import AnomalyDetectionFactory
import pandas as pd
import sys
import numpy as np
from Logger.logger import get_logger
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataConstructor
from Helpers.data_plotter import DataPlotter
from Helpers.data_creator import DataCreator
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)


def get_parameters(filename='params.yml'):
    params_helper = ParamsHelper(filename)
    ParamsValidator(params_helper).validate()
    return params_helper


def create_synthetic_data(create_data_params):
    filename = create_data_params.filename
    data_creator = DataCreator()
    df, anomalies_df = data_creator.create_data('2020-01-01 00:00', '2020-01-15 00:00', '10min')
    data_creator.save_to_csv(df, filename)
    return anomalies_df


def contruct_data(params_helper):
    metadata = params_helper.get_metadata()
    preprocess_data_params = params_helper.get_preprocess_data_params()
    data = DataConstructor(metadata, preprocess_data_params).read().build()
    return data


def run_experiment(params_helper, data):
    detector_name = params_helper.get_detector_name()
    experiment_hyperparameters = params_helper.get_experiment_hyperparams()
    model_hyperparameters = params_helper.get_model_hyperparams()
    detector = AnomalyDetectionFactory(detector_name, experiment_hyperparameters, model_hyperparameters) \
        .get_detector()

    logger = get_logger('run_experiment')
    logger.info("Starting experiment for anomaly detector: {}".format(detector_name))
    anomalies = detector.run_anomaly_detection(data)
    return anomalies


def output_results(params_helper, data, anomalies_pred_df, anomalies_true_df=pd.DataFrame()):
    is_output = params_helper.get_is_output()
    experiment_name = params_helper.create_experiment_name()

    if is_output.csv:
        anomalies_pred_df.to_csv(experiment_name + '.csv')

    if is_output.plot:
        DataPlotter.plot_anomalies(data=data,
                                   df_anomalies=anomalies_pred_df,
                                   plot_name=experiment_name,
                                   anomalies_true_df=anomalies_true_df)
    else:
        DataPlotter.plot_anomalies(data=data,
                                   df_anomalies=anomalies_pred_df,
                                   anomalies_true_df=anomalies_true_df)


if __name__ == "__main__":
    logger = get_logger('Main')
    logger.info('Starting')

    try:
        params_helper = get_parameters()

        create_data_params = params_helper.get_create_synthetic_data()
        anomalies_true_df = pd.DataFrame()
        if create_data_params.to_create:
            anomalies_true_df = create_synthetic_data(create_data_params)

        data = contruct_data(params_helper)
        anomalies_pred_df = run_experiment(params_helper, data)
        output_results(params_helper, data, anomalies_pred_df, anomalies_true_df)

    except Exception as e:
        logger.error(e)

    finally:
        logger.info('Finished')
