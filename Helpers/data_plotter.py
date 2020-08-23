import matplotlib.pyplot as plt
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
    def plot_anomalies(data, df_anomalies, plot_name=None, anomalies_true_df=pd.DataFrame()):
        plt.rcParams['date.epoch'] = '0000-12-31T00:00:00'  # workaround for date issue in plot

        if anomalies_true_df.empty:
            DataPlotter.plot_scatter_single_fig(data, df_anomalies, 'Predicted Anomalies')
        else:
            DataPlotter.plot_scatter_double_fig(data, df_anomalies, anomalies_true_df,
                                                'Predicted Anomalies', 'True Anomalies')

        if plot_name:
            plt.savefig(plot_name + '.jpg')

        plt.show()

    @staticmethod
    def plot_scatter_single_fig(df_plot, df_scatter=pd.DataFrame(), title=None):
        plt.plot(df_plot.iloc[:, 0], 'b')

        if df_scatter.shape[0]:
            x = df_scatter.index.values
            y = df_plot.loc[df_scatter.index].values.reshape(1, -1)
            plt.scatter(x=x, y=y, c='r')

        if title:
            plt.title(title)

    @staticmethod
    def plot_scatter_double_fig(df_plot, df_scatter_1=pd.DataFrame(), df_scatter_2=pd.DataFrame(),
                                title_1=None, title_2=None):
        fig, axs = plt.subplots(2)

        axs[0].plot(df_plot.iloc[:, 0], 'b')
        axs[1].plot(df_plot.iloc[:, 0], 'b')

        if df_scatter_1.shape[0]:
            x = df_scatter_1.index.values
            y = df_plot.loc[df_scatter_1.index].values.reshape(1, -1)
            axs[0].scatter(x=x, y=y, c='r')

            if title_1:
                axs[0].set_title(title_1)

        if df_scatter_1.shape[0]:
            x = df_scatter_2.index.values
            y = df_plot.loc[df_scatter_2.index].values.reshape(1, -1)
            axs[1].scatter(x=x, y=y, c='g')

            if title_2:
                axs[1].set_title(title_2)