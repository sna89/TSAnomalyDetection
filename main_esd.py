from Tasks.esd_task import ESDTask
from SeasonalESD.seasonal_esd import SeasonalESD
import pandas as pd
import sys
import numpy as np
from Logger.logger import create_logger
from Tasks.pre_process_task import PreProcessDataTask

if __name__ == "__main__":
    test = True
    if test:
        create_logger()
        pd.set_option('display.max_rows', None)
        np.set_printoptions(threshold=sys.maxsize)

        experiment_hyperparameters = dict()
        experiment_hyperparameters['train_period_weeks'] = 4
        experiment_hyperparameters['forecast_period_hours'] = 4
        experiment_hyperparameters['retrain_schedule_hours'] = 3
        attribute = 'internaltemp'

        model_hyperparameters = dict()
        model_hyperparameters['anomaly_ratio'] = 0.05
        model_hyperparameters['hybrid'] = True
        model_hyperparameters['alpha'] = 0.1

        filename = 'Sensor U106748.csv'

        pre_process_task = PreProcessDataTask(filename)
        data = pre_process_task.pre_process()

        esd_task = ESDTask(SeasonalESD, experiment_hyperparameters)
        df_anomalies = esd_task.run_experiment(data, model_hyperparameters, test=True)
        assert df_anomalies.shape[0] == 464, "Test of esd task failed"
        # esd_task.plot_seasonality_per_period(data)