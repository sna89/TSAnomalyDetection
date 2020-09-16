from Models.anomaly_detection_model import AnomalyDetectionModel
from fbprophet import Prophet
from Helpers.data_helper import DataHelper
import numpy as np


class FBProphet(AnomalyDetectionModel):
    def __init__(self, data, interval_width, forecast_period_hours):
        super(FBProphet, self).__init__(data)

        self.adjust_schema()
        self.interval_width = interval_width
        self.forecast_period_hours = forecast_period_hours

    def adjust_schema(self):
        col_name = self.data.name
        self.data = self.data.reset_index()
        self.data.rename(columns={'sampletime': 'ds', col_name: 'y'}, inplace=True)
        self.data.index = self.data.ds

    def run(self):
        model = Prophet(interval_width=self.interval_width, changepoint_prior_scale=0.2)
        train_df_raw, test_df_raw = DataHelper.split_train_test(self.data, self.forecast_period_hours)
        model.fit(train_df_raw)
        future = model.make_future_dataframe(periods=test_df_raw.shape[0], freq='10min')
        forecast = model.predict(future)
        forecast.index =  self.data.index
        forecast['actual'] = self.data['y']
        forecast['anomaly'] = np.where(
            (forecast.actual < forecast.yhat_lower) | (forecast.actual > forecast.yhat_upper), 1, 0)
        anomalies = forecast[forecast['anomaly'] == 1][['ds', 'actual']]
        anomalies.set_index('ds', drop=True, inplace=True)
        return anomalies