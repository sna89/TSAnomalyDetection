import six
import sys
from pmdarima.arima import auto_arima
from Models.model import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.modules['sklearn.externals.six'] = six


class Arima(Model):
    def __init__(self, data, seasonality):
        super(Arima, self).__init__(data)
        self.seasonality = seasonality
        self.fitted_model = None
        self.train_periods = 6*12
        self.test_periods = 6

    def fit(self):
        if not self.init:
            self.fitted_model = auto_arima(self.data[:self.train_periods], start_p=1, start_q=1,
                                         max_p=4, max_q=4, m=self.seasonality,
                                         seasonal=True,
                                         trace=True,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True)

            self.init = True

    def get_forecast(self, periods):
        if self.init:
            forecasts, conf_int = self.fitted_model.predict(periods, return_conf_int=True)
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
        self.fit()
        forecasts, conf_int = self.get_forecast()

        summary_df = pd.DataFrame(data={'forecasts': forecasts,
                                        'lower_limit_conf_int': conf_int[:, 0],
                                        'upper_limit_conf_int': conf_int[:, 1]},
                                  index=self.data[self.train_periods:self.train_periods + self.test_periods].index)

        summary_df['is_anomaly'] = np.where((summary_df.forecasts < summary_df.lower_limit_conf_int) |
                                            (summary_df.forecasts > summary_df.upper_limit_conf_int),
                                            1,
                                            0)

        return summary_df[summary_df['is_anomaly'] == 1]


