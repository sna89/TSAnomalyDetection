from AnomalyDetectors.ad_factory import AnomalyDetectionFactory
import pandas as pd
import sys
import numpy as np
from Logger.logger import create_logger
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataBuilder

if __name__ == "__main__":
    create_logger()
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)

    try:
        params_helper = ParamsHelper('arima_params.yml')
        ParamsValidator(params_helper).validate()

        metadata = params_helper.get_metadata()
        data_builder = DataBuilder(metadata)
        data = data_builder.build()

        detector_name = params_helper.get_detector_name()
        experiment_hyperparameters = params_helper.get_experiment_hyperparams()
        model_hyperparameters = params_helper.get_model_hyperparams()
        detector = AnomalyDetectionFactory(detector_name, experiment_hyperparameters, model_hyperparameters).get_detector()

        is_test = params_helper.get_test()
        detector.run_anomaly_detection(data)


        # DataPlotter.plot_anomalies(data, df_anomalies)
    except Exception as e:
        print(e)
