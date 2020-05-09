from Tasks.task import ESDTask
from seasonal_esd import SeasonalESD
import pandas as pd

pd.set_option('display.max_rows', None)

filename = 'Sensor U106748.csv'

anomaly_ratio = 0.05
hybrid = True
alpha = 0.01

esd_task = ESDTask(SeasonalESD)
esd_task.run_experiment(filename, anomaly_ratio, alpha, hybrid)