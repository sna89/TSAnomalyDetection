from Tasks.esd_task import ESDTask
from seasonal_esd import SeasonalESD
import pandas as pd
import sys
import numpy as np
from Logger.logger import create_logger

create_logger()
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)

filename = 'Sensor U106748.csv'

experiment_hyperparameters = dict()
experiment_hyperparameters['train_period_weeks'] = 4
experiment_hyperparameters['forecast_period_hours'] = 3
experiment_hyperparameters['retrain_schedule_hours'] = 3
attribute = 'internalrh'
esd_task = ESDTask(SeasonalESD, attribute, experiment_hyperparameters)

model_hyperparameters = dict()
model_hyperparameters['anomaly_ratio'] = 0.05
model_hyperparameters['hybrid'] = False
model_hyperparameters['alpha'] = 0.1

esd_task.run_experiment(filename, model_hyperparameters, test=False)
# esd_task.plot_seasonality_per_period(filename)