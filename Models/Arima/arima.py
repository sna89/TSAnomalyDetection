import six
import sys
from pmdarima.arima import auto_arima
from Models.anomaly_detection_model import AnomalyDetectionModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Helpers.data_helper import DataHelper
sys.modules['sklearn.externals.six'] = six


class Arima(AnomalyDetectionModel):
    def __init__(self, data, seasonality, forecast_periods_hours):
        super(Arima, self).__init__(data)
        self.seasonality = seasonality
        self.fitted_model = None

        self.train_df, self.test_df = DataHelper.split_train_test(self.data, forecast_periods_hours)
        self.train_periods = self.train_df.shape[0]
        self.test_periods = self.test_df.shape[0]

    def fit(self):
        if not self.init:
            self.fitted_model = auto_arima(self.train_df, start_p=1, start_q=1,
                                         max_p=4, max_q=4, m=self.seasonality,
                                         seasonal=True,
                                         trace=True,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True)

            self.logger.info("Chosen Arima model: {}*{}".
                             format(self.fitted_model.order, self.fitted_model.seasonal_order))

            self.init = True

    def get_forecast(self):
        if self.init:
            forecasts, conf_int = self.fitted_model.predict(self.test_periods, return_conf_int=True)
            return forecasts, conf_int
        else:
            msg = "Need to fit arima model in order to get forecast"
            self.logger.error(msg)

    def show_model_summary(self):
        if self.init:
            print(self.fitted_model.summary())
        else:
            msg = "Need to fit arima model in order to show summary"
            self.logger.debug(msg)

    def plot_diagnostics(self):
        if self.init:
            self.fitted_model.plot_diagnostics(figsize=(15, 12))
            plt.show()
        else:
            msg = "Need to fit arima model in order to plot diagnostics"
            self.logger.debug(msg)

    def run(self):
        if not self.is_constant_train_data():
            self.fit()
            forecasts, conf_int = self.get_forecast()

            summary_df = pd.DataFrame(data={'forecasts': forecasts,
                                            'lower_limit_conf_int': conf_int[:, 0],
                                            'upper_limit_conf_int': conf_int[:, 1]},
                                      index=self.data[self.train_periods:].index)

            summary_df['is_anomaly'] = np.where((summary_df.forecasts < summary_df.lower_limit_conf_int) |
                                                (summary_df.forecasts > summary_df.upper_limit_conf_int),
                                                1,
                                                0)

            return summary_df[summary_df['is_anomaly'] == 1]
        else:
            return pd.DataFrame()

    def plot_forecast(self):
        if self.init:
            self.show_model_summary()
            forecasts, conf_int = self.get_forecast()

            plt.plot(self.train_df.index, self.train_df.values, alpha=0.75)
            plt.plot(self.test_df.index, forecasts, alpha=0.75)  # Forecasts
            plt.scatter(self.test_df.index, self.test_df.values,
                        alpha=0.4, marker='x')  # Test data
            plt.fill_between(self.test_df.index,
                             conf_int[:, 0], conf_int[:, 1],
                             alpha=0.1, color='b')

            plt.show()
        else:
            msg = "Need to fit arima model in order to plot forecast"
            self.logger.debug(msg)

    def is_constant_train_data(self):
        unique_values = self.train_df.nunique().values[0]
        return True if unique_values == 1 else False

