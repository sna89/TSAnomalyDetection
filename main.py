import pandas as pd
import sys
import numpy as np
from Tasks.sign_task import SignTask
from Pilfer.sign import Sign
from Tasks.pre_process_task import PreProcessDataTask
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)

#Sensor U100256 dehydrator
filenames = ['Sensor U95696.csv', 'Sensor U100256.csv', 'Sensor U100310.csv', 'Sensor U106748.csv']

experiment_hyperparameters = dict()
experiment_hyperparameters['train_period_weeks'] = 1
experiment_hyperparameters['forecast_period_hours'] = 0
experiment_hyperparameters['retrain_schedule_hours'] = 3

model_hyperparameters = dict()
model_hyperparameters['alpha'] = 0.5

pre_process_task = PreProcessDataTask(*filenames)
data = pre_process_task.pre_process()



data.plot()
plt.show()

sign_task = SignTask(Sign, experiment_hyperparameters)
sign_task.run_experiment(data, model_hyperparameters, test=False, scale=True)