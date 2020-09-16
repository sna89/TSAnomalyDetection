from AnomalyDetectors.ad import AnomalyDetector
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from dataclasses import dataclass
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter


@dataclass
class SeasonalESDHyperParameters:
    anomaly_ratio: float
    hybrid: bool
    alpha: float


class ESDAnomalyDetector(AnomalyDetector):
    def __init__(self, model, experiment_hyperparameters, model_hyperparameters):
        super(ESDAnomalyDetector, self).__init__(model, experiment_hyperparameters, model_hyperparameters)
        # self.esd_hyperparameters = SeasonalESDHyperParameters(**model_hyperparameters)

    # def fit_model(self, df_):
    #     model = self.model(df_,
    #                        self.esd_hyperparameters.anomaly_ratio,
    #                        self.esd_hyperparameters.alpha,
    #                        self.esd_hyperparameters.hybrid)
    #     self.fitted_model = model.fit()

    # def detect_anomalies(self):
    #     return self.fitted_model.detect()

    def plot_seasonality_per_period(self, data, freq='M'):
        periods = DataHelper.split(data, freq)
        for idx, period in enumerate(periods):
            period = period.iloc[:, 0]
            result = seasonal_decompose(period, model='additive', period=30)
            result.plot()
            median = period.median()
            self.logger.info("period {} median: {}".format(idx+1, median))
            resid = pd.Series(data=period - result.seasonal - median, name='resid')
            resid.plot()
            plt.savefig('{}_seasonal_decompose.jpg'.format(idx+1))
            plt.clf()
            DataPlotter.qqplot(resid, show=False)
            plt.savefig('{}_qqplot.jpg'.format(idx+1))


