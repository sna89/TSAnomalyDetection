import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import statsmodels.api as sm
import pylab as py
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataPlotter:
    def __init__(self):
        sns.set()

    @staticmethod
    def plot_data_distribution(data):
        sns.distplot(data)
        plt.show()

    @staticmethod
    def plot_ts_data(data):
        fig = go.Figure()

        for col in data.columns:
            fig.add_trace(go.Scatter(x=data[col].index, y=data[col].values,
                                     mode='lines',
                                     name=col))
        fig.write_html("TSData.html")

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
    def plot_anomalies(data, df_anomalies, anomalies_true_df=pd.DataFrame()):
        # plt.rcParams['date.epoch'] = '0000-12-31T00:00:00'  # workaround for date issue in plot

        if anomalies_true_df.empty:
            DataPlotter.plot_scatter_single_fig(data, df_anomalies, 'Predicted Anomalies')
        else:
            DataPlotter.plot_scatter_double_fig(data, df_anomalies, anomalies_true_df,
                                                'Predicted Anomalies', 'True Anomalies')

    @staticmethod
    def plot_scatter_single_fig(df_plot, df_scatter=pd.DataFrame(), title=None):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_plot.iloc[:, 0].index, y=df_plot.iloc[:, 0].values,
                                 mode='lines',
                                 name='Temperature'))

        if df_scatter.shape[0]:
            x = df_scatter.index
            y = df_plot.loc[df_scatter.index].values.reshape(1, -1)[0]
            fig.add_trace(go.Scatter(y=y,
                                     x=x,
                                     mode='markers',
                                     name='Anomalies'))

        fig.update_layout(title=title,
                          xaxis_title='Date',
                          yaxis_title='Temperature')

        fig.write_html("anomalies.html")

    @staticmethod
    def plot_scatter_double_fig(df_plot, df_scatter_1=pd.DataFrame(), df_scatter_2=pd.DataFrame(),
                                trace_1=None, trace_2=None, title=None):
        fig = make_subplots(rows=2, cols=1)

        fig.append_trace(go.Scatter(x=df_plot.iloc[:, 0].index,
                                    y=df_plot.iloc[:, 0].values,
                                    mode='lines',
                                    name='Temperature'),
                         row=1, col=1)

        fig.append_trace(go.Scatter(x=df_plot.iloc[:, 0].index,
                                    y=df_plot.iloc[:, 0].values,
                                    mode='lines',
                                    name='Temperature'),
                         row=2, col=1)

        if df_scatter_1.shape[0]:
            x = df_scatter_1.index
            y = df_plot.loc[df_scatter_1.index].values.reshape(1, -1)[0]
            fig.append_trace(go.Scatter(y=y,
                                        x=x,
                                        mode='markers',
                                        name=trace_1),
                             row=1, col=1)

        if df_scatter_1.shape[0]:
            x = df_scatter_2.index
            y = df_plot.loc[df_scatter_2.index].values.reshape(1, -1)[0]
            fig.append_trace(go.Scatter(y=y,
                                        x=x,
                                        mode='markers',
                                        name=trace_2),
                             row=2, col=1)

        fig.update_layout(title_text=title)

        fig.write_html("anomalies.html")
