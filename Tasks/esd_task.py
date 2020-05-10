from Logger.logger import MethodLogger
from dateutil.relativedelta import relativedelta
from seasonal_esd import SeasonalESDHyperParameters
import copy
from Tasks.task import Task, ExperimentHyperParameters
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from time import time

ATTRIBUTES = ['internaltemp', 'internalrh']


class ESDTask(Task):
    def __init__(self, model, attribute, experiment_hyperparameters):
        super(ESDTask, self).__init__(model)
        assert attribute in ATTRIBUTES, 'Attribute must be "internaltemp" or "internalrh"'
        self.attribute = attribute
        self.df_anomalies = pd.DataFrame()
        self.experiment_hyperparameters = ExperimentHyperParameters(**experiment_hyperparameters)
        self.model_hyperparameters = None

    @MethodLogger
    def run_experiment(self, filename, model_hyperparameters, test=True):
        start = time()

        self.model_hyperparameters = SeasonalESDHyperParameters(**model_hyperparameters)

        self.read_data(filename)
        df_raw = self.pre_process_data()

        first_obs_time, last_obs_time = self.get_first_and_last_times(df_raw)
        start_time, end_time = self.init_train_period(first_obs_time)
        if test:
            df_raw = df_raw.loc[start_time:start_time + relativedelta(weeks=12)] #total time
            last_obs_time = df_raw.index.max()
            end_time = start_time + relativedelta(weeks=4) # train time
        df = copy.deepcopy(df_raw)

        while end_time <= last_obs_time:
            df_ = pd.DataFrame(data=copy.deepcopy(df.loc[start_time:end_time]),
                               columns=[self.attribute])

            detected_anomalies = self.detect_anomalies(df_)
            filtered_anomalies = self.filter_anomalies_in_forecast(end_time, detected_anomalies)

            if not filtered_anomalies.empty:
                self.df_anomalies = pd.concat([self.df_anomalies, filtered_anomalies], axis=0)
                self.logger.info("Filtered anomalies using ESD between {0} - {1}: {2}"
                                 .format(start_time, end_time, filtered_anomalies))
            start_time, end_time = self.update_train_period(start_time, end_time, last_obs_time)
            df = df.drop(labels=detected_anomalies.index, axis=0)

            del df_

        self.data_plotter.plot_anomalies(df_raw, self.df_anomalies)

        end = time()
        self.logger.info("Total runtime of esd task for attribute {0}: {1} minutes"
                         .format(self.attribute, (end - start)/float(60)))

    def pre_process_data(self):
        data = self.data_helper.pre_process(self.data, index='Time', pivot_column='Type', value_columns=['Value'])
        data = data['Value'][self.attribute]
        return data

    @staticmethod
    def time_in_range(current, start, end):
        assert start <= end, 'start time must be ealier than end time'
        current = current.at.obj.name
        # current = pd.to_datetime(current)
        return start <= current <= end

    def detect_anomalies(self, df_):
        esd_model = self.model(df_,
                               self.model_hyperparameters.anomaly_ratio,
                               self.model_hyperparameters.alpha,
                               self.model_hyperparameters.hybrid)

        detected_anomalies = esd_model.run()
        if not detected_anomalies.empty:
            detected_anomalies = pd.DataFrame(data=detected_anomalies,
                                              columns=[self.attribute],
                                              index=detected_anomalies.index)
        return detected_anomalies

    def filter_anomalies_in_forecast(self, end_time, detected_anomalies):
        filtered_anomalies = pd.DataFrame()
        if not detected_anomalies.empty:
            forecast_period = end_time + relativedelta(hours=-self.experiment_hyperparameters.forecast_period_hours)
            detected_anomalies['forecast'] = detected_anomalies.apply(
                self.time_in_range, axis=1, args=(forecast_period, end_time,))
            filtered_anomalies = detected_anomalies[detected_anomalies['forecast'] == True][self.attribute]
        return filtered_anomalies

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

    @staticmethod
    def get_first_and_last_times(df_):
        first_obs_time = df_.index.min()
        last_obs_time = df_.index.max()
        return first_obs_time, last_obs_time

    def plot_seasonality_per_period(self, filename, freq='M'):
        self.read_data(filename)
        df_raw = self.pre_process_data()
        periods = self.data_helper.split_data(df_raw, freq)
        for idx, period in enumerate(periods):
            result = seasonal_decompose(period, model='additive', period=30)
            result.plot()
            median = period.median()
            self.logger.info("period {} median: {}".format(idx+1, median))
            resid = pd.Series(data=period - result.seasonal - median, name='resid')
            resid.plot()
            plt.savefig('{}_seasonal_decompose.jpg'.format(idx+1))

            plt.clf()
            self.data_plotter.qqplot(resid, show=False)
            plt.savefig('{}_qqplot.jpg'.format(idx+1))


