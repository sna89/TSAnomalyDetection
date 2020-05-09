from abc import ABC, abstractmethod
import pandas as pd
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter
from Logger.logger import MethodLogger
from dateutil.relativedelta import relativedelta
from Logger.logger import create_logger
import copy
import numpy as np

ATTRIBUTES = ['internaltemp', 'internalrh']


class Task(ABC):
    def __init__(self, model):
        assert model is not None, 'Need to pass a class model'
        self.model = model

        self.data_helper = DataHelper()
        self.data_plotter = DataPlotter()
        create_logger()

        self.data = None

    def run(self):
        assert hasattr(self.model, 'run'), 'Model must implement "run" function'
        self.model.run()

    def read_data(self, filename):
        self.data = pd.read_csv(filename)

    @abstractmethod
    def pre_process_data(self):
        pass


class ESDTask(Task):
    def __init__(self, model):
        super(ESDTask, self).__init__(model)
        self.df_anomalies = pd.DataFrame()

    @MethodLogger
    def run_experiment(self, filename,  *parameters):
        self.read_data(filename)
        df_raw = self.pre_process_data()
        df = copy.deepcopy(df_raw)
        start_time = df.index.min()
        df = df.loc[start_time:start_time+relativedelta(weeks=4)]
        df_raw = df_raw.loc[start_time:start_time+relativedelta(weeks=4)]
        end_time = start_time + relativedelta(weeks=1)
        while end_time < df.index.max():
            df_ = pd.DataFrame(data=copy.deepcopy(df['Value'][ATTRIBUTES[0]].loc[start_time:end_time]),
                               columns=[ATTRIBUTES[0]])
            esd_model = self.model(df_, *parameters)
            detected_anomalies = esd_model.run()
            detected_anomalies = pd.DataFrame(data=detected_anomalies,
                                              columns=[ATTRIBUTES[0]],
                                              index=detected_anomalies.index)

            if not detected_anomalies.empty:
                forecast_period = end_time + relativedelta(hours=-6)
                detected_anomalies['forecast'] = detected_anomalies.apply(
                   self.time_in_range, axis=1, args=(forecast_period, end_time,))
                detected_anomalies = detected_anomalies[detected_anomalies['forecast']==True][ATTRIBUTES[0]]
                self.df_anomalies = pd.concat([self.df_anomalies, detected_anomalies], axis=0)
                df = df.drop(labels=detected_anomalies.index, axis=0)

            start_time = start_time + relativedelta(hours=6)
            end_time = end_time + relativedelta(hours=6)

            del df_

        self.data_plotter.plot_anomalies(df_raw['Value'][ATTRIBUTES[0]], self.df_anomalies)

    def pre_process_data(self):
        return self.data_helper.pre_process(self.data, index='Time', pivot_column='Type', value_columns=['Value'])

    @staticmethod
    def time_in_range(current, start, end):
        assert start <= end, 'start time must be ealier than end time'
        current = current.at.obj.name
        # current = pd.to_datetime(current)
        return start <= current <= end