from dateutil.relativedelta import relativedelta
from SeasonalESD.seasonal_esd import SeasonalESDHyperParameters
from Tasks.task import Task, ExperimentHyperParameters
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


class ESDTask(Task):
    def __init__(self, model, experiment_hyperparameters, attribute='internaltemp'):
        super(ESDTask, self).__init__(model, experiment_hyperparameters, attribute)

    def get_model_hyperparameters(self, model_hyperparameters):
        self.model_hyperparameters = SeasonalESDHyperParameters(**model_hyperparameters)

    @staticmethod
    def time_in_range(current, start, end):
        assert start <= end, 'start time must be earlier than end time'
        return (start <= current.index) & (current.index <= end)

    def detect_anomalies(self, df_):
        model = self.model(df_,
                           self.model_hyperparameters.anomaly_ratio,
                           self.model_hyperparameters.alpha,
                           self.model_hyperparameters.hybrid)

        return self.run(model)

    def filter_anomalies_in_forecast(self, end_time, detected_anomalies):
        forecast_period = end_time + relativedelta(hours=-self.experiment_hyperparameters.forecast_period_hours)
        filtered = pd.Series(self.time_in_range(detected_anomalies, forecast_period, end_time),
                             index=detected_anomalies.index,
                             name='is_filtered')
        filtered = pd.concat([detected_anomalies, filtered], axis=1)
        filtered = filtered[filtered['is_filtered']==True]
        return filtered[filtered.columns[0]]

    def plot_seasonality_per_period(self, data, freq='M'):
        periods = self.data_helper.split_data(data, freq)
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
            self.data_plotter.qqplot(resid, show=False)
            plt.savefig('{}_qqplot.jpg'.format(idx+1))


