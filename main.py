from Tasks.esd_task import ESDTask
from seasonal_esd import SeasonalESD
import pandas as pd
import sys
import numpy as np

pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)

filename = 'Sensor U106748.csv'

experiment_hyperparameters = dict()
experiment_hyperparameters['train_period_weeks'] = 4
experiment_hyperparameters['forecast_period_hours'] = 3
experiment_hyperparameters['retrain_schedule_hours'] = 3
attribute = 'internaltemp'
esd_task = ESDTask(SeasonalESD, attribute, experiment_hyperparameters)

model_hyperparameters = dict()
model_hyperparameters['anomaly_ratio'] = 0.01
model_hyperparameters['hybrid'] = False
model_hyperparameters['alpha'] = 0.05

esd_task.run_experiment(filename, model_hyperparameters, test=False)