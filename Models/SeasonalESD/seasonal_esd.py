import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import math
from scipy import stats
from Helpers.data_plotter import DataPlotter
from Logger.logger import MethodLogger
from dataclasses import dataclass
from Models.model import Model

@dataclass
class SeasonalESDHyperParameters:
    anomaly_ratio: float
    hybrid: bool
    alpha: float


class SeasonalESD(Model):
    def __init__(self, data, anomaly_ratio, alpha, hybrid=False):
        super(SeasonalESD, self).__init__(data)
        assert anomaly_ratio <= 0.49, "anomaly ratio is too high"

        self.data_plotter = DataPlotter()

        self.nobs = self.data.shape[0]
        self.k = math.ceil(float(self.nobs) * anomaly_ratio)

        self.hybrid = hybrid
        self.alpha = alpha

        self.indices = None

    @MethodLogger
    def run(self):
        if not self.init:
            self.init = True
            resid = self._get_updated_resid()
            self.indices = self._esd(resid)

            if not self.indices.empty:
                self.anomaly_df = self.data.loc[self.indices.values]
                return self.anomaly_df.sort_index()
            else:
                return pd.Series()
        else:
            print("Already executed Seasonal-ESD algorithm")

    def _get_seasonal_decomposition(self, model='additive', period=7):
        result = seasonal_decompose(self.data, model=model, period=period)
        return result

    def _get_updated_resid(self):
        median = self.data.median()
        result = self._get_seasonal_decomposition()
        resid = pd.Series(data=self.data - result.seasonal - median, name='resid')
        return resid

    def _esd(self, resid):
        critical_values = self._calc_critical_values()
        indices, statistics = self._calc_statistics(resid)

        self.summary_df = pd.DataFrame(data={'statistic': statistics,
                                             'critical_value': critical_values},
                                       index=indices)
        self.summary_df['anomaly'] = np.where(self.summary_df.statistic > self.summary_df.critical_value, 1, 0)
        indices = self._get_predicted_anomalies_indices()
        return indices

    def _calc_critical_values(self):
        critical_values = []
        for i in range(1, self.k+1):
            df = self.nobs-i-1
            p = 1-self.alpha/(2*(self.nobs-i+1))
            t_stat = stats.t.ppf(p, df=df)

            numerator = (self.nobs-i)*t_stat
            denominator = np.sqrt((self.nobs-i-1+t_stat**2)*(self.nobs-i+1))
            critical_value = numerator/denominator

            critical_values.append(critical_value)

        return critical_values

    def _calc_statistics(self, df):
        indices = []
        statistics = []
        for i in range(1, self.k + 1):
            idx, statistic = self._calc_statistic(df)
            statistics.append(statistic)
            indices.append(idx)

            df = df.drop(axis=0, labels=idx)
        return indices, statistics

    def _calc_statistic(self, df):
        if self.hybrid:
            median = df.median()
            mad = self.calc_mad(df, median)
            statistic_df = np.abs(df - median) / mad
        else:
            mean = df.mean()
            std = df.std()
            statistic_df = np.abs(df - mean) / std

        idx = statistic_df.idxmax(axis=0)
        statistic = statistic_df.max(axis=0)
        return idx, statistic

    @staticmethod
    def calc_mad(df, median=None):
        if not median:
            median = df.median()
        mad = np.abs(df-median).median()
        return mad

    def get_summary_df(self):
        if self.init:
            self.summary_df.set_index(keys='index', drop=True, inplace=True)
            self.summary_df.drop(labels=['level_0'], axis=1, inplace=True)
            self.summary_df['value'] = self.data.loc[self.indices.values]
            return self.summary_df
        else:
            print('Need to call run() first')

    def _get_predicted_anomalies_indices(self):
        self.summary_df = self.summary_df.reset_index()
        if self.summary_df['anomaly'].sum() == 0:
            return pd.Series()
        self.summary_df = self.summary_df.reset_index()
        max_idx_anomaly = self.summary_df[self.summary_df['anomaly'] == 1]['level_0'].idxmax(axis=0)
        indices = self.summary_df[:max_idx_anomaly+1]['index']
        return indices

    def plot_residual_distribution(self):
        resid = self._get_updated_resid()
        self.data_plotter.plot_data_distribution(resid)
        self.data_plotter.qqplot(resid)