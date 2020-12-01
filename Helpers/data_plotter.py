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
            DataPlotter.plot_scatter_multiple_trace(data, df_anomalies, anomalies_true_df,
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
    def plot_scatter_multiple_trace(df_plot, df_scatter_1=pd.DataFrame(), df_scatter_2=pd.DataFrame(),
                                    trace_1=None, trace_2=None, title=None):

        num_features = df_plot.shape[1]

        fig = go.Figure()

        for feature in range(num_features):
            fig.add_trace(go.Scatter(x=df_plot.iloc[:, feature].index,
                                     y=df_plot.iloc[:, feature].values,
                                     mode='lines',
                                     name='Value',
                                     marker=dict(color=feature)))

        if df_scatter_1.shape[0]:
            for feature in range(num_features):
                x = df_scatter_1.index
                y = df_plot.loc[df_scatter_1.index].values[:, feature]
                fig.add_trace(go.Scatter(y=y,
                                         x=x,
                                         mode='markers',
                                         name=trace_1 + '_feature_{}'.format(feature + 1),
                                         marker=dict(color=0)))

                x = df_scatter_2.index
                y = df_plot.loc[df_scatter_2.index].values[:, feature]
                fig.add_trace(go.Scatter(y=y,
                                         x=x,
                                         mode='markers',
                                         name=trace_2 + '_feature_{}'.format(feature + 1),
                                         marker=dict(color=1)))

        fig.update_layout(title_text=title)
        fig.write_html("anomalies.html")

    @staticmethod
    def plot_double_fig(df_1, df_2, title):
        assert df_1.shape[1] == df_2.shape[1], "dataframes must have same number of features"
        num_features = df_1.shape[1]

        fig = make_subplots(rows=num_features, cols=1)
        for feature in range(num_features):
            fig.append_trace(go.Scatter(x=df_1.iloc[:, feature].index,
                                        y=df_1.iloc[:, feature].values,
                                        mode='lines',
                                        name='Value'),
                             row=1, col=1)
            fig.append_trace(go.Scatter(x=df_2.iloc[:, feature].index,
                                        y=df_2.iloc[:, feature].values,
                                        mode='lines',
                                        name='Value'),
                             row=2, col=1)

        fig.update_layout(title_text=title)
        fig.write_html("{}.html".format(title))
