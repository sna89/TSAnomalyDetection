from AnomalyDetectors.ad_factory import AnomalyDetectionFactory
import pandas as pd
import sys
import numpy as np
from Logger.logger import create_logger
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataBuilder
from Helpers.data_plotter import DataPlotter

if __name__ == "__main__":
    create_logger()
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)

    test = True
    if test:
        try:
            params_helper = ParamsHelper('esd_params.yml')
            params_validator = ParamsValidator(params_helper)

            metadata = params_helper.get_metadata()
            data_builder = DataBuilder(metadata)
            data = data_builder.build()

            detector_type = params_helper.get_detector_type()
            experiment_hyperparameters = params_helper.get_experiment_hyperparams()
            model_hyperparameters = params_helper.get_model_hyperparams()
            detector = AnomalyDetectionFactory(detector_type, experiment_hyperparameters, model_hyperparameters).get_detector()

            df_anomalies = detector.run_anomaly_detection(data, test=True)
            DataPlotter.plot_anomalies(data, df_anomalies)
        except Exception as e:
            print(e)
