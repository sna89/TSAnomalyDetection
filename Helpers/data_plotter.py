from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
import pandas as pd
import statsmodels.api as sm
import pylab as py
import seaborn as sns


class DataPlotter:
    def __init__(self):
        sns.set()

    @staticmethod
    def plot_data_distribution(data):
        sns.distplot(data)
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
    def qqplot(data, show=True):
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        sm.qqplot(data, line='45')
        if show:
            py.show()

    @staticmethod
    def plot_anomalies(df, df_anomalies):
        if df_anomalies.shape[0]:
            plt.plot(df.iloc[:, 0], 'b')
            x = df_anomalies.index.values
            y = df.loc[df_anomalies.index].values.reshape(1, -1)
            plt.scatter(x=x, y=y, c='r')
            plt.show()

