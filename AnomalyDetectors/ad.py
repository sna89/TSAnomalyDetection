from abc import ABC, abstractmethod
import pandas as pd
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter
from Logger.logger import get_logger
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
from time import time
from typing import Dict
import copy


@dataclass
class TrainPeriod:
    hours: int
    days: int
    weeks: int


@dataclass
class ExperimentHyperParameters:
    train_period: Dict
    forecast_period_hours: int
    retrain_schedule_hours: int


class AnomalyDetector(ABC):
    def __init__(self, model, experiment_hyperparameters):
        assert model is not None, 'Need to pass a class model'
        self.data_helper = DataHelper()
        self.data_plotter = DataPlotter()
        self.logger = get_logger()

        self.df_anomalies = pd.DataFrame()
        self.model = model

        self.experiment_hyperparameters = ExperimentHyperParameters(**experiment_hyperparameters)
        self.train_period = TrainPeriod(**self.experiment_hyperparameters.train_period)

    @staticmethod
    def run_model(model):
        assert hasattr(model, 'run'), 'Model must implement "run" function'
        return model.run()

    @staticmethod
    def test(df_raw, start_time):
        end_time = DataHelper.get_max_idx(df_raw, start_time + relativedelta(weeks=8))
        df_raw = df_raw.loc[start_time:end_time]
        last_obs_time = df_raw.index.max()
        return df_raw, last_obs_time

    def run_anomaly_detection(self, data, test=True, scale=False):
        self.logger.info("Start running anomaly detection experiment")
        start = time()

        first_obs_time, last_obs_time = DataHelper.get_first_and_last_observations(data)
        epoch_start_time, epoch_end_time = self.init_train_period(data, first_obs_time)

        if test:
            data, last_obs_time = AnomalyDetector.test(data, first_obs_time)

        if scale:
            data = DataHelper.scale(data)

        df_no_anomalies = copy.deepcopy(data)

        while epoch_end_time <= last_obs_time:
            self.logger.info('Detecting anomalies between {} to {}'.format(epoch_start_time, epoch_end_time))

            df_ = pd.DataFrame(data=copy.deepcopy(df_no_anomalies.loc[epoch_start_time:epoch_end_time]))
            detected_anomalies = self.detect_anomalies(df_)

            if not detected_anomalies.empty:
                filtered_anomalies = self.filter_anomalies_in_forecast(detected_anomalies, epoch_end_time)

                if not filtered_anomalies.empty:
                    self.logger.info("Filtered anomalies: {}".format(filtered_anomalies))
                    self.df_anomalies = pd.concat([self.df_anomalies, filtered_anomalies], axis=0)
                else:
                    self.logger.info("No anomalies detected")

                df_no_anomalies.drop(labels=detected_anomalies.index, axis=0, inplace=True)

            else:
                self.logger.info("No anomalies detected")

            epoch_start_time, epoch_end_time = self.update_train_period(df_no_anomalies,
                                                                        epoch_start_time,
                                                                        epoch_end_time,
                                                                        last_obs_time)
            del df_

        end = time()
        self.logger.info("Total runtime of anomaly detection experiment: {0} minutes"
                         .format((end - start) / float(60)))

        return self.df_anomalies

    def update_train_period(self, df, start_time, end_time, last_obs_time):
        updated_start_time = DataHelper.get_min_idx(df,
                                                    start_time +
                                                    relativedelta(hours=self.experiment_hyperparameters.retrain_schedule_hours))
        updated_end_time = end_time + relativedelta(hours=self.experiment_hyperparameters.retrain_schedule_hours)

        if updated_end_time <= last_obs_time:
            updated_end_time = DataHelper.get_min_idx(df, updated_end_time)

        return updated_start_time, updated_end_time

    def init_train_period(self, data, first_obs_time):
        epoch_start_time = first_obs_time
        epoch_end_time = DataHelper.get_max_idx(data, first_obs_time
                                                + relativedelta(hours=self.train_period.hours)
                                                + relativedelta(days=self.train_period.days)
                                                + relativedelta(weeks=self.train_period.weeks))
        return epoch_start_time, epoch_end_time

    def filter_anomalies_in_forecast(self, detected_anomalies, forecast_end_time):
        forecast_start_time = forecast_end_time - \
                              relativedelta(hours=self.experiment_hyperparameters.forecast_period_hours)
        filtered = pd.Series(DataHelper.time_in_range(detected_anomalies, forecast_start_time, forecast_end_time),
                             index=detected_anomalies.index,
                             name='is_filtered')
        filtered = pd.concat([detected_anomalies, filtered], axis=1)
        filtered = filtered[filtered['is_filtered'] == True]
        return filtered

    @abstractmethod
    def detect_anomalies(self, df_):
        pass
