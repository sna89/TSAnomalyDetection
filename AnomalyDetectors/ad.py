import pandas as pd
from Helpers.data_helper import DataHelper, Period, timer
from Logger.logger import get_logger
from Helpers.params_helper import ExperimentHyperParameters
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from constants import AnomalyDfColumns
from Helpers.time_freq_converter import TimeFreqConverter


class AnomalyDetector():
    def __init__(self, model, experiment_hyperparameters, model_hyperparameters, freq, categorical_columns):
        assert model is not None, 'Need to pass a class model'
        self.logger = get_logger(__class__.__name__)

        self.df_anomalies = pd.DataFrame()
        self.model = model

        self.experiment_hyperparameters = ExperimentHyperParameters(**experiment_hyperparameters)

        self.train_period = Period(**self.experiment_hyperparameters.train_period)
        self.train_period_num_samples = TimeFreqConverter.convert_to_num_samples(self.train_period, freq=freq)

        train_freq = Period(**self.experiment_hyperparameters.train_freq)
        self.train_freq_delta = timedelta(hours=train_freq.minutes) + \
                                timedelta(hours=train_freq.hours) + \
                                timedelta(days=train_freq.days) + \
                                timedelta(weeks=train_freq.weeks)

        self.forecast_period = Period(**self.experiment_hyperparameters.forecast_period)
        self.forecast_period_num_samples = TimeFreqConverter.convert_to_num_samples(self.forecast_period, freq=freq)

        self.model_hyperparameters = model_hyperparameters
        self.model_hyperparameters['forecast_period'] = self.forecast_period_num_samples
        self.model_hyperparameters['categorical_columns'] = categorical_columns
        self.model_hyperparameters['freq'] = freq
        self.model_hyperparameters['input_timesteps_period'] = TimeFreqConverter.convert_to_num_samples(
            Period(**model_hyperparameters['input_timesteps_period']), freq=freq
        )

    @timer
    def run_anomaly_detection_experiment(self, data_):
        self.logger.info("Start running anomaly detection experiment")

        model = self.model(self.model_hyperparameters)
        elapsed_time = timedelta(hours=0)
        to_fit = True

        data = data_.copy()
        for epoch, idx in enumerate(range(0,
                                          len(data) - (self.train_period_num_samples + self.forecast_period_num_samples),
                                          self.forecast_period_num_samples),
                                    start=1):
            df_curr_epoch = data.iloc[idx: idx + self.train_period_num_samples + self.forecast_period_num_samples]

            detection_start_time = df_curr_epoch.iloc[-self.forecast_period_num_samples:].index.min()
            detection_end_time = df_curr_epoch.index.max()
            self.logger.info('Epoch: {}, Detecting anomalies between {} to {}'.
                             format(epoch, detection_start_time, detection_end_time))

            if elapsed_time >= self.train_freq_delta:
                del model
                model = self.model(self.model_hyperparameters)
                elapsed_time = timedelta(hours=0)
                to_fit = True

            if to_fit:
                model = model.fit(df_curr_epoch)
                to_fit = False

            detected_anomalies = model.detect(df_curr_epoch)

            if not detected_anomalies.empty:
                filtered_anomalies = detected_anomalies

                if epoch > 1 or not self.experiment_hyperparameters.include_train_time:
                    filtered_anomalies = self.filter_anomalies_in_forecast(detected_anomalies, df_curr_epoch.index.max())

                if not filtered_anomalies.empty:
                    self.df_anomalies = pd.concat([self.df_anomalies, filtered_anomalies], axis=0)
                    self.logger.info("Filtered anomalies: {}".format(filtered_anomalies))
                else:
                    self.logger.info("No anomalies detected in current iteration")

                if self.experiment_hyperparameters.remove_outliers:
                    if AnomalyDfColumns.Prediction in detected_anomalies.columns:
                        actual_detected_anomalies = detected_anomalies[detected_anomalies[AnomalyDfColumns.IsAnomaly] == 1]
                        for idx, row in actual_detected_anomalies.iterrows():
                            feature = row[AnomalyDfColumns.Feature]
                            prediction = row[AnomalyDfColumns.Prediction]
                            actual = row[AnomalyDfColumns.Actual]
                            df_curr_epoch.at[idx, feature] = prediction * 0.5 + actual * (1 - 0.5)

                    else:
                        df_curr_epoch.drop(labels=detected_anomalies.index, axis=0, inplace=True)

            else:
                self.logger.info("No anomalies detected in current iteration")

            elapsed_time += timedelta(minutes=self.forecast_period.minutes,
                                      hours=self.forecast_period.hours,
                                      days=self.forecast_period.days,
                                      weeks=self.forecast_period.weeks)

        return self.df_anomalies

    def filter_anomalies_in_forecast(self, detected_anomalies, forecast_end_time):
        forecast_start_time = forecast_end_time - \
                              relativedelta(minutes=self.forecast_period.minutes,
                                            hours=self.forecast_period.hours,
                                            days=self.forecast_period.days,
                                            weeks=self.forecast_period.weeks)
        filtered = pd.Series(DataHelper.time_in_range(detected_anomalies, forecast_start_time, forecast_end_time),
                             index=detected_anomalies.index,
                             name='is_filtered')
        filtered = pd.concat([detected_anomalies, filtered], axis=1)
        filtered = filtered[filtered['is_filtered'] == True]
        filtered.drop(columns=['is_filtered'], axis=1, inplace=True)
        return filtered


