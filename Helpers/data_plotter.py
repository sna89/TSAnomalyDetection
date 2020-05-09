from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
import pandas as pd
import statsmodels.api as sm
import pylab as py


class DataPlotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_data_distribution(data, kernel='gaussian'):
        x = data.values.reshape(-1, 1)
        int_max_x = int(np.max(x))
        x_plot = np.linspace(0, int_max_x, int_max_x * 2)[:, np.newaxis]

        kde = KernelDensity(kernel=kernel, bandwidth=0.3).fit(x)
        logprob = kde.score_samples(x_plot)

        plt.fill_between(x_plot.reshape(1, -1)[0], np.exp(logprob), alpha=0.5)
        plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
        plt.xlim(np.min(x), np.max(x))
        plt.show()

    @classmethod
    def plot_ts_data(cls, data):
        cls._update_figure_size(17, 8)
        cls.plot_data(data)
        cls._update_figure_size(6.4, 4.8)

    @staticmethod
    def plot_data(data):
        plt.plot(data, 'b')
        plt.show()

    @staticmethod
    def _update_figure_size(width, height):
        rcParams['figure.figsize'] = width, height

    @staticmethod
    def qqplot(data):
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        sm.qqplot(data, line='45')
        py.show()

    @staticmethod
    def plot_anomalies(df, df_anomalies):
        if df_anomalies.shape[0]:
            plt.plot(df, 'b')
            x = df_anomalies.index.values
            y = df_anomalies.values.reshape(1, -1)[0]
            plt.scatter(x=x, y=y, c='r')
            plt.show()

