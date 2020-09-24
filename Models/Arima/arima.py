import six
import sys
from pmdarima.arima import auto_arima
from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Helpers.data_helper import DataHelper
from statsmodels.tsa.statespace.sarimax import SARIMAX
sys.modules['sklearn.externals.six'] = six


ARIMA_HYPERPARAMETERS = ['seasonality', 'forecast_period_hours']


class Arima(AnomalyDetectionModel):
    def __init__(self, model_hyperparameters):
        super(Arima, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(ARIMA_HYPERPARAMETERS, model_hyperparameters)
        self.seasonality = model_hyperparameters['seasonality']
        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']


        self.fitted = False
        self.fitted_model = None

    def _init_data(self, data):
        self.data = AnomalyDetectionModel.init_data(data)

    def fit(self, data):
        self._init_data(data)
        train_df = AnomalyDetectionModel.get_train_set(data, self.forecast_period_hours)
        if not DataHelper.is_constant_data(train_df):
            self.fitted_model = auto_arima(train_df, start_p=1, start_q=1,
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
            self.logger.info("Constant train data in Arima model")

    @validate_anomaly_df_schema
    def detect(self, data):
        self._init_data(data)
        train_df, test_df = DataHelper.split_train_test(data, self.forecast_period_hours)

        if not DataHelper.is_constant_data(train_df):
            model = self._fit_chosen_model(train_df)

            test_periods = test_df.shape[0]
            forecasts, conf_int = self._get_forecast(model, test_periods)

            # self._plot_forecast(model, test_periods, train_df, test_df)

            summary_df = pd.DataFrame(data={'actual': test_df.values.reshape(forecasts.shape),
                                            'forecasts': forecasts.values,
                                            'lower_limit_conf_int': conf_int.iloc[:, 0].values,
                                            'upper_limit_conf_int': conf_int.iloc[:, 1].values
                                            },
                                      index=data[train_df.shape[0]:].index)

            summary_df['is_anomaly'] = np.where((summary_df.actual < summary_df.lower_limit_conf_int) |
                                                (summary_df.actual > summary_df.upper_limit_conf_int),
                                                1,
                                                0)

            return summary_df[summary_df['is_anomaly'] == 1]['actual']
        else:
            return pd.Series()

    def _fit_chosen_model(self, train_df):
        if self.fitted:
            model = SARIMAX(train_df,
                            order=self.fitted_model.order,
                            seasonal_order=self.fitted_model.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model = model.fit(disp=False)
            return model
        else:
            raise Exception("Need to fit arima model in order to get fit current train data")

    def _get_forecast(self, model, periods):
        if self.fitted:
            forecasts = model.get_forecast(steps=periods, return_conf_int=True)

            predictions = forecasts.predicted_mean
            conf_int = forecasts.conf_int()

            return predictions, conf_int
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

    def _plot_forecast(self, model, periods, train_df, test_df):
        if self.fitted:
            forecasts, conf_int = self._get_forecast(model, periods)

            # plt.plot(train_df.index, train_df.values, alpha=0.75)
            plt.plot(test_df.index, forecasts, alpha=0.75)  # Forecasts
            plt.scatter(test_df.index, test_df.values,
                        alpha=0.4, marker='x')  # Test data
            plt.fill_between(test_df.index,
                             conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                             alpha=0.1, color='b')


        else:
            msg = "Need to fit arima model in order to plot forecast"
            self.logger.debug(msg)

