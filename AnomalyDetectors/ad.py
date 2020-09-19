import pandas as pd
from Helpers.data_helper import DataHelper, Period, timer
from Logger.logger import get_logger
from dataclasses import dataclass
from typing import Dict
import copy
from dateutil.relativedelta import relativedelta
from datetime import timedelta


@dataclass
class ExperimentHyperParameters:
    train_period: Dict
    train_freq: Dict
    forecast_period_hours: int
    scale: bool


class AnomalyDetector():
    def __init__(self, model, experiment_hyperparameters, model_hyperparameters=None):
        assert model is not None, 'Need to pass a class model'
        self.logger = get_logger(__class__.__name__)

        self.df_anomalies = pd.DataFrame()
        self.model = model

        self.experiment_hyperparameters = ExperimentHyperParameters(**experiment_hyperparameters)
        self.train_period = Period(**self.experiment_hyperparameters.train_period)

        train_freq = Period(**self.experiment_hyperparameters.train_freq)
        self.train_freq_delta = timedelta(hours=train_freq.hours) + \
                                timedelta(days=train_freq.days) + \
                                timedelta(weeks=train_freq.weeks)

        self.model_hyperparameters = model_hyperparameters
        self.model_hyperparameters['forecast_period_hours'] = self.experiment_hyperparameters.forecast_period_hours

    @timer
    def run_anomaly_detection_experiment(self, data):
        self.logger.info("Start running anomaly detection experiment")

        last_obs_time = AnomalyDetector._get_last_observations_time(data)
        epoch_start_time, epoch_end_time = self._init_train_period(data)

        df_no_anomalies = copy.deepcopy(data)
        df_ = pd.DataFrame(data=copy.deepcopy(df_no_anomalies.loc[epoch_start_time:epoch_end_time]))
        model = self.model(self.model_hyperparameters)
        epoch = 1
        elapsed_time = timedelta(hours=0)
        fit = True

        while epoch_end_time <= last_obs_time:
            self.logger.info('Epoch: {}, '
                             'Detecting anomalies between {} to {}'.
                             format(epoch,
                                    epoch_end_time - relativedelta(hours=self.experiment_hyperparameters.forecast_period_hours),
                                    epoch_end_time))

            if elapsed_time >= self.train_freq_delta:
                del model
                model = self.model(self.model_hyperparameters)
                elapsed_time = timedelta(hours=0)
                fit = True

            if fit:
                model = model.fit(df_)
                fit = False

            detected_anomalies = model.detect(df_)

            if not detected_anomalies.empty:
                filtered_anomalies = self.filter_anomalies_in_forecast(detected_anomalies, epoch_end_time)

                if not filtered_anomalies.empty:
                    self.logger.info("Filtered anomalies: {}".format(filtered_anomalies))
                    self.df_anomalies = pd.concat([self.df_anomalies, filtered_anomalies], axis=0)
                else:
                    self.logger.info("No anomalies detected in current iteration")

                df_no_anomalies.drop(labels=detected_anomalies.index, axis=0, inplace=True)

            else:
                self.logger.info("No anomalies detected in current iteration")

            epoch_start_time, epoch_end_time = self._update_train_period(df_no_anomalies,
                                                                         epoch_start_time,
                                                                         epoch_end_time,
                                                                         last_obs_time)
            del df_
            df_ = pd.DataFrame(data=copy.deepcopy(df_no_anomalies.loc[epoch_start_time:epoch_end_time]))
            elapsed_time += timedelta(hours=self.experiment_hyperparameters.forecast_period_hours)
            epoch += 1

        return self.df_anomalies

    def _update_train_period(self, df, start_time, end_time, last_obs_time):
        updated_start_time = DataHelper.get_min_idx(df,
                                                    start_time +
                                                    relativedelta(hours=self.experiment_hyperparameters.forecast_period_hours))
        updated_end_time = end_time + relativedelta(hours=self.experiment_hyperparameters.forecast_period_hours)

        if updated_end_time <= last_obs_time:
            updated_end_time = DataHelper.get_min_idx(df, updated_end_time)

        return updated_start_time, updated_end_time

    def _init_train_period(self, data):
        epoch_start_time = data.index.min()
        epoch_end_time = DataHelper.get_max_idx(data, DataHelper.relative_delta_time(epoch_start_time,
                                                                                     minutes=0,
                                                                                     hours=self.train_period.hours,
                                                                                     days=self.train_period.days,
                                                                                     weeks=self.train_period.weeks))
        return epoch_start_time, epoch_end_time

    @staticmethod
    def _get_last_observations_time(df):
        last_obs_time = df.index.max()
        return last_obs_time

    def filter_anomalies_in_forecast(self, detected_anomalies, forecast_end_time):
        forecast_start_time = forecast_end_time - \
                              relativedelta(hours=self.experiment_hyperparameters.forecast_period_hours)
        filtered = pd.Series(DataHelper.time_in_range(detected_anomalies, forecast_start_time, forecast_end_time),
                             index=detected_anomalies.index,
                             name='is_filtered')
        filtered = pd.concat([detected_anomalies, filtered], axis=1)
        filtered = filtered[filtered['is_filtered'] == True]
        filtered.drop(columns=['is_filtered'], axis=1, inplace=True)
        return filtered

    # @abstractmethod
    # def fit_model(self, df_):
    #     pass
    #
    # @abstractmethod
    # def detect_anomalies(self):
    #     pass


