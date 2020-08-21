from AnomalyDetectors.ad_factory import AnomalyDetectionFactory
import pandas as pd
import sys
import numpy as np
from Logger.logger import create_logger, get_logger
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataConstructor
from Helpers.data_plotter import DataPlotter
from Helpers.data_creator import DataCreator

if __name__ == "__main__":
    create_logger()
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)

    try:
        params_helper = ParamsHelper()
        ParamsValidator(params_helper).validate()

        create_data = params_helper.get_create_data()
        if create_data.create:
            filename = create_data.filename
            data_creator = DataCreator()
            df = data_creator.create_data('2020-01-01 00:00', '2020-01-08 00:00', '10min')
            data_creator.save_to_csv(df, filename)

        metadata = params_helper.get_metadata()
        preprocess_data_params = params_helper.get_preprocess_data_params()

        data = DataConstructor(metadata, preprocess_data_params).read().build()

        detector_name = params_helper.get_detector_name()
        experiment_hyperparameters = params_helper.get_experiment_hyperparams()
        model_hyperparameters = params_helper.get_model_hyperparams()
        detector = AnomalyDetectionFactory(detector_name, experiment_hyperparameters, model_hyperparameters)\
            .get_detector()

        anomalies = detector.run_anomaly_detection(data)

        is_output = params_helper.get_is_output()
        experiment_name = params_helper.create_experiment_name()

        if is_output.csv:
            anomalies.to_csv(experiment_name + '.csv')

        if is_output.plot:
            DataPlotter.plot_anomalies(data, anomalies, experiment_name)
        else:
            DataPlotter.plot_anomalies(data, anomalies)

    except Exception as e:
        logger = get_logger('main')
        logger.error(e)
