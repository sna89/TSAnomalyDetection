import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
# import statsmodels.api as sm
# import pylab as py
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from constants import Paths, AnomalyDfColumns


class DataPlotter:
    def __init__(self):
        sns.set()

    @staticmethod
    def plot_data_distribution(data):
        sns.distplot(data)
        plt.show()

    @staticmethod
    def plot_ts_data(data, plot_name="TSData.html"):
        fig = go.Figure()

        for col in data.columns:
            fig.add_trace(go.Scatter(x=data[col].index, y=data[col].values,
                                     mode='lines',
                                     name=col))

        plot_file_path = os.path.join(Paths.output_path, plot_name)
        fig.write_html(plot_file_path)

    @staticmethod
    def plot_data(data):
        plt.plot(data, 'b')
        plt.show()

    @staticmethod
    def _update_figure_size(width, height):
        rcParams['figure.figsize'] = width, height

    # @staticmethod
    # def qqplot(data, show=True):
    #     if isinstance(data, pd.DataFrame):
    #         data = data.iloc[:, 0]
    #     sm.qqplot(data, line='45')
    #     if show:
    #         py.show()

    @staticmethod
    def plot_anomalies(data, predicted_anomaly_df, actual_anomaly_df=pd.DataFrame(), features_to_plot=None):
        # plt.rcParams['date.epoch'] = '0000-12-31T00:00:00'  # workaround for date issue in plot
        DataPlotter.plot_scatter_single_fig(data, predicted_anomaly_df, actual_anomaly_df, features_to_plot, 'Predicted Anomalies')

    @staticmethod
    def plot_scatter_single_fig(data_df,
                                predicted_anomaly_df,
                                actual_anomaly_df,
                                features_to_plot=None,
                                title='Predicted Anomalies'):

        features = data_df.columns
        if not features_to_plot and isinstance(features_to_plot, list):
            features = features_to_plot

        for feature in features:
            fig = go.Figure()

            actual = data_df[feature].values
            fig.add_trace(go.Scatter(x=data_df.index, y=actual,
                                     mode='lines',
                                     line=go.scatter.Line(color="blue"),
                                     name=feature
                                     ))

            if not predicted_anomaly_df.empty:
                feature_anomaly_df = predicted_anomaly_df[predicted_anomaly_df[AnomalyDfColumns.Feature] == feature]

                index = pd.DatetimeIndex(feature_anomaly_df.index)

                predictions = feature_anomaly_df[AnomalyDfColumns.Prediction]
                fig.add_trace(
                    go.Scatter(
                        x=index,
                        y=predictions,
                        mode="lines",
                        line=go.scatter.Line(dash='dash', color="red"),
                        name='Prediction')
                )

                if AnomalyDfColumns.LowerBound in predicted_anomaly_df.columns and \
                        AnomalyDfColumns.UpperBound in predicted_anomaly_df.columns:

                    lower_bound = feature_anomaly_df[AnomalyDfColumns.LowerBound]
                    fig.add_trace(
                        go.Scatter(
                            x=index,
                            y=lower_bound,
                            mode="lines",
                            line=go.scatter.Line(color="lightskyblue"),
                            name='Lower bound')
                    )

                    upper_bound = feature_anomaly_df[AnomalyDfColumns.UpperBound]
                    fig.add_trace(
                        go.Scatter(
                            x=index,
                            y=upper_bound,
                            mode="lines",
                            line=go.scatter.Line(color="lightskyblue"),
                            fill='tonexty',
                            name='Upper bound')
                    )

                elif AnomalyDfColumns.Distance in predicted_anomaly_df.columns:
                    pass

                else:
                    raise ValueError("Anomaly predictions dataframe does not"
                                     "contain lower/upper bound or distance metric")

                feature_predicted_anomaly_df = feature_anomaly_df[feature_anomaly_df[AnomalyDfColumns.IsAnomaly] == 1]
                feature_predicted_anomaly_df_index = pd.DatetimeIndex(feature_predicted_anomaly_df.index)

                actual_values_for_anomaly_indices = data_df[feature].loc[feature_predicted_anomaly_df_index].values
                index_for_anomaly_indices = data_df[feature].loc[feature_predicted_anomaly_df_index].index

                fig.add_trace(go.Scatter(y=actual_values_for_anomaly_indices,
                                         x=index_for_anomaly_indices,
                                         mode='markers',
                                         marker=dict(
                                             color='brown',
                                             size=8),
                                         name='Predicted Anomalies'))

                if not actual_anomaly_df.empty:
                    actual_anomalies_index = pd.DatetimeIndex(actual_anomaly_df.index)
                    actual_anomalies_values = data_df.loc[actual_anomalies_index]

                    fig.add_trace(go.Scatter(y=actual_anomalies_values,
                                             x=actual_anomalies_index,
                                             mode='markers',
                                             marker=dict(
                                                 color='green',
                                                 size=8),
                                             name='True Anomalies'))

                fig.update_layout(title=feature + ' ' + title,
                                  xaxis_title='Date',
                                  yaxis_title=feature)

                fig.write_html(os.path.join(Paths.output_path, "anomalies_{}.html".format(feature)))

    @staticmethod
    def plot_scatter_double_fig(df_plot, df_scatter_1=pd.DataFrame(), df_scatter_2=pd.DataFrame(),
                                trace_1=None, trace_2=None, title=None):

        num_features = df_plot.shape[1]

        fig = make_subplots(rows=num_features, cols=1)

        for feature in range(num_features):
            fig.append_trace(go.Scatter(x=df_plot.iloc[:, feature].index,
                                        y=df_plot.iloc[:, feature].values,
                                        mode='lines',
                                        name='Temperature'),
                             row=feature + 1, col=1)

        if df_scatter_1.shape[0]:
            for feature in range(num_features):
                x = df_scatter_1.index
                y = df_plot.loc[df_scatter_1.index].values[:, feature]
                fig.append_trace(go.Scatter(y=y,
                                            x=x,
                                            mode='markers',
                                            name=trace_1 + '_feature_{}'.format(feature + 1),
                                            marker=dict(color=0)),
                                 row=feature + 1, col=1)

                x = df_scatter_2.index
                y = df_plot.loc[df_scatter_2.index].values[:, feature]
                fig.append_trace(go.Scatter(y=y,
                                            x=x,
                                            mode='markers',
                                            name=trace_2 + '_feature_{}'.format(feature + 1),
                                            marker=dict(color=1)),
                                 row=feature + 1, col=1)

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
