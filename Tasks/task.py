from abc import ABC, abstractmethod
import pandas as pd
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter
from Logger.logger import get_logger
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
from Logger.logger import MethodLogger
from time import time
import copy
from sklearn import preprocessing


@dataclass
class ExperimentHyperParameters:
    train_period_weeks: int
    forecast_period_hours: int
    retrain_schedule_hours: int


class Task(ABC):
    def __init__(self, model, experiment_hyperparameters, attribute='internaltemp'):
        assert model is not None, 'Need to pass a class model'
        self.data_helper = DataHelper()
        self.data_plotter = DataPlotter()
        self.logger = get_logger()

        self.df_anomalies = pd.DataFrame()
        self.model = model
        self.attribute = attribute

        self.experiment_hyperparameters = ExperimentHyperParameters(**experiment_hyperparameters)
        self.model_hyperparameters = None

    @staticmethod
    def run(model):
        assert hasattr(model, 'run'), 'Model must implement "run" function'
        return model.run()

    @staticmethod
    def read_data(filename):
        return pd.read_csv(filename)

    @staticmethod
    def get_first_and_last_observations(df_):
        first_obs_time = df_.index.min()
        last_obs_time = df_.index.max()
        return first_obs_time, last_obs_time

    @staticmethod
    def test(df_raw, start_time):
        df_raw = df_raw.loc[start_time:start_time + relativedelta(weeks=4)]  # total time
        last_obs_time = df_raw.index.max()
        end_time = start_time + relativedelta(weeks=3)  # train time
        return df_raw, last_obs_time, end_time

    @MethodLogger
    def run_experiment(self, data, model_hyperparameters, test=True, scale=False):
        start = time()

        self.get_model_hyperparameters(model_hyperparameters)

        first_obs_time, last_obs_time = Task.get_first_and_last_observations(data)
        start_time, end_time = self.init_train_period(first_obs_time)

        if test:
            data, last_obs_time, end_time = Task.test(data, start_time)

        raw_data = copy.deepcopy(data)

        if scale:
            scaler = preprocessing.StandardScaler()
            scaler.fit(data)
            data[data.columns] = scaler.transform(data)

        df = copy.deepcopy(data)

        while end_time <= last_obs_time:
            df_ = pd.DataFrame(data=copy.deepcopy(df.loc[start_time:end_time]))

            detected_anomalies = self.detect_anomalies(df_)
            if not detected_anomalies.empty:
                filtered_anomalies = self.filter_anomalies_in_forecast(end_time, detected_anomalies)

                if not filtered_anomalies.empty:
                    self.df_anomalies = pd.concat([self.df_anomalies, filtered_anomalies], axis=0)
                    self.logger.info("Filtered anomalies using ESD between {0} - {1}: {2}"
                                     .format(start_time, end_time, filtered_anomalies))

                df = df.drop(labels=detected_anomalies.index, axis=0)

            start_time, end_time = self.update_train_period(start_time, end_time, last_obs_time)
            del df_

        self.data_plotter.plot_anomalies(raw_data, self.df_anomalies)

        end = time()
        self.logger.info("Total runtime of esd task for attribute {0}: {1} minutes"
                         .format(self.attribute, (end - start) / float(60)))

        return self.df_anomalies

    def update_train_period(self, start_time, end_time, last_obs_time):
        updated_start_time = start_time + relativedelta(hours=self.experiment_hyperparameters.retrain_schedule_hours)
        updated_end_time = end_time + relativedelta(hours=self.experiment_hyperparameters.retrain_schedule_hours)
        if end_time < last_obs_time:
            if updated_end_time >= last_obs_time:
                updated_end_time = last_obs_time
        return updated_start_time, updated_end_time

    def init_train_period(self, first_obs_time):
        start_time = first_obs_time
        end_time = start_time + relativedelta(weeks=self.experiment_hyperparameters.train_period_weeks)
        return start_time, end_time

    def get_model_hyperparameters(self, model_hyperparameters):
        pass

    def detect_anomalies(self, df_):
        pass

    def filter_anomalies_in_forecast(self, end_time, detected_anomalies):
        pass

