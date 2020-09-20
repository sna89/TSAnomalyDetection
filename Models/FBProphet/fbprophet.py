from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from fbprophet import Prophet
from Helpers.data_helper import DataHelper
import numpy as np


PROPHET_HYPERPARAMETERS = ['interval_width', 'changepoint_prior_scale', 'forecast_period_hours']


class FBProphet(AnomalyDetectionModel):
    def __init__(self, model_hyperparameters):
        super(FBProphet, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(PROPHET_HYPERPARAMETERS, model_hyperparameters)
        self.interval_width = model_hyperparameters['interval_width']
        self.changepoint_prior_scale = model_hyperparameters['changepoint_prior_scale']
        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']

        self.model = Prophet(interval_width=self.interval_width, changepoint_prior_scale=self.changepoint_prior_scale)

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

    def fit(self, df):
        df = self.init_data(df)
        train_df_raw, test_df_raw = DataHelper.split_train_test(df, self.forecast_period_hours)
        self.model = self.model.fit(train_df_raw)
        return self

    @validate_anomaly_df_schema
    def detect(self, df):
        df = self.init_data(df)
        # train_df_raw, test_df_raw = DataHelper.split_train_test(df, self.forecast_period_hours)
        # future = self.model.make_future_dataframe(periods=test_df_raw.shape[0], freq='10min')
        forecast = self.model.predict(df)
        forecast.index = df.index
        forecast['actual'] = df['y']
        forecast['anomaly'] = np.where(
            (forecast.actual < forecast.yhat_lower) | (forecast.actual > forecast.yhat_upper), 1, 0)
        anomalies = forecast[forecast['anomaly'] == 1]['actual']
        return anomalies



