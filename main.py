from AnomalyDetectors.ad_factory import AnomalyDetectionFactory
import pandas as pd
import sys
import numpy as np
from Logger.logger import get_logger
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataConstructor
from Builders.eval_builder import EvalBuilder
from Helpers.data_plotter import DataPlotter
from Helpers.data_creator import DataCreator, DataCreatorConst
import warnings
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)


def get_and_validate_parameters(filename='params.yml'):
    params_helper = ParamsHelper(filename)
    ParamsValidator(params_helper).validate()
    return params_helper


def create_synthetic_data(synthetic_data_params):
    filename = synthetic_data_params.filename
    data_creator = DataCreator()
    df, anomalies_df = data_creator.create_data(DataCreatorConst.START_DATE,
                                                DataCreatorConst.END_DATE,
                                                DataCreatorConst.FREQ)
    data_creator.save_to_csv(df, filename)
    return df, anomalies_df


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
    anomalies = detector.run_anomaly_detection_experiment(data)
    return anomalies


def output_results(params_helper, data, anomalies_pred_df, anomalies_true_df=pd.DataFrame()):
    is_output = params_helper.get_is_output()
    experiment_name = params_helper.create_experiment_name()

    if is_output.csv:
        anomalies_pred_df.to_csv('2 day train.csv')

    if is_output.plot:
        DataPlotter.plot_anomalies(data=data,
                                   df_anomalies=anomalies_pred_df,
                                   plot_name=experiment_name,
                                   anomalies_true_df=anomalies_true_df)
    else:
        DataPlotter.plot_anomalies(data=data,
                                   df_anomalies=anomalies_pred_df,
                                   anomalies_true_df=anomalies_true_df)

    evaluate_experiment(data, anomalies_pred_df, anomalies_true_df)


def evaluate_experiment(data, anomalies_pred_df, anomalies_true_df=pd.DataFrame()):
    if anomalies_true_df.empty:
        return
    else:
        evaluator = EvalBuilder(data, anomalies_true_df, anomalies_pred_df)
        evaluator.build()
        evaluator.output_confusion_matrix()
        evaluator.output_classification_report()
        evaluator.output_auc()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    logger = get_logger('Main')
    logger.info('Starting')
    try:
        params_helper = get_and_validate_parameters()
        synthetic_data_params = params_helper.get_synthetic_data_params()
        anomalies_true_df = pd.DataFrame()

        if synthetic_data_params.to_create:
            _, anomalies_true_df = create_synthetic_data(synthetic_data_params)

        anomalies = params_helper.get_anomalies()
        if anomalies:
            anomalies_true_df = pd.read_csv(anomalies,
                                            index_col=['Unnamed: 0'])
            anomalies_true_df.index = pd.to_datetime(anomalies_true_df.index)

        data = contruct_data(params_helper)
        # DataPlotter.plot_ts_data(data)
        data_1 = data.loc[:'11/11/2019']
        data_2 = data.loc['18/11/2019':]
        for data in [data_1, data_2]:
            anomalies_pred_df = run_experiment(params_helper, data)
            output_results(params_helper, data, anomalies_pred_df, anomalies_true_df)

    except Exception as e:
        logger.error(e)

    finally:
        logger.info('Finished')
