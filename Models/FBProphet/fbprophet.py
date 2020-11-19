from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from fbprophet import Prophet
from Helpers.data_helper import DataHelper
import numpy as np
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric, plot_plotly, plot_components_plotly
from fbprophet.diagnostics import cross_validation, performance_metrics


PROPHET_HYPERPARAMETERS = ['interval_width',
                           'changepoint_prior_scale',
                           'forecast_period_hours',
                           'daily_seasonality',
                           'weekly_seasonality',
                           'holidays_country_name']


class FBProphet(AnomalyDetectionModel):
    def __init__(self, model_hyperparameters):
        super(FBProphet, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(PROPHET_HYPERPARAMETERS, model_hyperparameters)

        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']

        self.interval_width = model_hyperparameters['interval_width']
        self.changepoint_prior_scale = model_hyperparameters['changepoint_prior_scale']
        self.daily_seasonality = model_hyperparameters['daily_seasonality']
        self.weekly_seasonality = model_hyperparameters['weekly_seasonality']
        self.holidays_country_name = model_hyperparameters['holidays_country_name']

        self.model = self.create_new_model()

    def create_new_model(self):
        model = Prophet(interval_width=self.interval_width,
                        changepoint_prior_scale=self.changepoint_prior_scale,
                        daily_seasonality=self.daily_seasonality,
                        weekly_seasonality=self.weekly_seasonality)

        # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        if self.holidays_country_name:
            model.add_country_holidays(country_name=self.holidays_country_name)
        return model

    @staticmethod
    def _adjust_prophet_schema(data):
        col_name = data.name

        data = data.reset_index()
        time_columns = data.columns[0]

        data.rename(columns={time_columns: 'ds', col_name: 'y'}, inplace=True)
        data.index = data.ds
        return data

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)
        data = FBProphet._adjust_prophet_schema(data)
        return data

    @staticmethod
    def _remove_3_std_outliers(train_df):
        train_mean = train_df.mean()[0]
        train_std = train_df.std()[0]
        train_df['3_std_anomaly'] = np.where(
            (train_df.y < train_mean - 3 * train_std) | (train_df.y > train_mean + 3 * train_std), 1, 0)
        train_df = train_df[train_df['3_std_anomaly'] == 0][['ds', 'y']]
        return train_df

    def _cross_validation(self):
        df_cv = cross_validation(self.model, horizon='7 days', period='1 days', initial='14 days', parallel="processes")
        df_p = performance_metrics(df_cv)
        plot_cross_validation_metric(df_cv, metric='mse')
        plot_cross_validation_metric(df_cv, metric='mae')
        plt.show()
        plt.close()

    def fit(self, df):
        df = self.init_data(df)
        train_df_raw, test_df_raw = DataHelper.split_train_test(df, self.forecast_period_hours)
        train_df = self._remove_3_std_outliers(train_df_raw)
        self.model = self.model.fit(train_df)
        # self._cross_validation()
        return self

    @validate_anomaly_df_schema
    def detect(self, df):
        df = self.init_data(df)

        # train_df_raw, test_df_raw = DataHelper.split_train_test(df, self.forecast_period_hours)
        # future = self.model.make_future_dataframe(periods=test_df_raw.shape[0], freq='10min')

        forecast = self.model.predict(df)

        self._plot_forecast(forecast)
        self._plot_components(forecast)

        forecast.index = df.index
        forecast['actual'] = df['y']
        forecast['anomaly'] = np.where(
            (forecast.actual < forecast.yhat_lower) | (forecast.actual > forecast.yhat_upper), 1, 0)

        anomalies = forecast[forecast['anomaly'] == 1][['actual', 'yhat_lower', 'yhat_upper']]
        return anomalies

    def _plot_forecast(self, forecast):
        fig = plot_plotly(self.model, forecast)
        fig.write_html("prophet_forecast.html")

    def _plot_components(self, forecast):
        fig = plot_components_plotly(self.model, forecast)
        fig.write_html("prophet_components.html")


