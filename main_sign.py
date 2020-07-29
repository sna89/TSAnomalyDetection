import pandas as pd
import sys
import numpy as np
from AnomalyDetectors.sign_ad import SignAnomalyDetector
from Pilfer.sign import Sign
from AnomalyDetectors.pre_process_task import PreProcessTask
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)

    #Sensor U100256 dehydrator
    filenames = ['Sensor U95696.csv', 'Sensor U100256.csv', 'Sensor U100310.csv', 'Sensor U106748.csv']

    experiment_hyperparameters = dict()
    experiment_hyperparameters['train_period_weeks'] = 1
    experiment_hyperparameters['forecast_period_hours'] = 0
    experiment_hyperparameters['retrain_schedule_hours'] = 3

    model_hyperparameters = dict()
    model_hyperparameters['alpha'] = 0.05

    pre_process_task = PreProcessTask(*filenames)
    data = pre_process_task.pre_process()

    data.plot()
    plt.show()

    sign_task = SignAnomalyDetector(Sign, experiment_hyperparameters)
    sign_task.run_anomaly_detection(data, model_hyperparameters, test=False, scale=True)