import six
import sys
from pmdarima.arima import auto_arima
from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema, validate_data, clean_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Helpers.data_helper import DataHelper
from Logger.logger import get_logger
sys.modules['sklearn.externals.six'] = six


class Arima(AnomalyDetectionModel):
    def __init__(self, seasonality, forecast_periods_hours):
        super(Arima, self).__init__()

        self.seasonality = seasonality
        self.forecast_periods_hours = forecast_periods_hours

        self.fitted = False
        self.constant = False
        self.fitted_model = None
        self.logger = get_logger(__class__.__name__)

    def _init_data(self, data):
        self.data = AnomalyDetectionModel.init_data(data)
        self.train_df, self.test_df = DataHelper.split_train_test(self.data, self.forecast_periods_hours)
        self.train_periods = self.train_df.shape[0]
        self.test_periods = self.test_df.shape[0]

    def fit(self, data):
        self._init_data(data)
        if not self.is_constant_train_data():
            self.fitted_model = auto_arima(self.train_df, start_p=1, start_q=1,
                                         max_p=4, max_q=4, m=self.seasonality,
                                         seasonal=True,
                                         trace=True,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True)

            self.fitted = True
            self.logger.info("Chosen Arima model: {}*{}".
                             format(self.fitted_model.order, self.fitted_model.seasonal_order))
            return self
        else:
            self.constant = True
            self.logger.info("Constant train data in Arima model")

    @validate_anomaly_df_schema
    def detect(self, data):
        if not self.constant:
            forecasts, conf_int = self.get_forecast()

            summary_df = pd.DataFrame(data={'actual': self.test_df,
                                            'forecasts': forecasts,
                                            'lower_limit_conf_int': conf_int[:, 0],
                                            'upper_limit_conf_int': conf_int[:, 1]},
                                      index=data[self.train_periods:].index)

            summary_df['is_anomaly'] = np.where((summary_df.forecasts < summary_df.lower_limit_conf_int) |
                                                (summary_df.forecasts > summary_df.upper_limit_conf_int),
                                                1,
                                                0)

            return summary_df[summary_df['is_anomaly'] == 1]
        else:
            return pd.Series()

    def get_forecast(self):
        if self.fitted:
            forecasts, conf_int = self.fitted_model.predict(self.test_periods, return_conf_int=True)
            return forecasts, conf_int
        else:
            msg = "Need to fit arima model in order to get forecast"
            self.logger.error(msg)

    def show_model_summary(self):
        if self.fitted:
            print(self.fitted_model.summary())
        else:
            msg = "Need to fit arima model in order to show summary"
            self.logger.debug(msg)

    def plot_diagnostics(self):
        if self.fitted:
            self.fitted_model.plot_diagnostics(figsize=(15, 12))
            plt.show()
        else:
            msg = "Need to fit arima model in order to plot diagnostics"
            self.logger.debug(msg)

    def plot_forecast(self):
        if self.fitted:
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

