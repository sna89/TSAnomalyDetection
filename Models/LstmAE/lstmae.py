from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras
import pandas as pd
import numpy as np
from Helpers.data_helper import DataHelper, DataConst
from sklearn.metrics import mean_squared_error
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import normalize
from Helpers.data_plotter import DataPlotter

pd.options.mode.chained_assignment = None


LSTMAE_HYPERPARAMETERS = ['hidden_layer', 'dropout', 'threshold', 'forecast_period_hours']


class LstmAE(AnomalyDetectionModel):
    def __init__(self, model_hyperparameters):
        super(LstmAE, self).__init__()

        AnomalyDetectionModel.validate_model_hyperpameters(LSTMAE_HYPERPARAMETERS, model_hyperparameters)
        self.hidden_layer = model_hyperparameters['hidden_layer']
        self.dropout = model_hyperparameters['dropout']
        self.batch_size = model_hyperparameters['batch_size']
        self.threshold = model_hyperparameters['threshold']
        self.forecast_period_hours = model_hyperparameters['forecast_period_hours']

        self.model = None

    @staticmethod
    def prepare_data_lstm(data, forecast_period_hours):
        forecast_samples = int(forecast_period_hours * 6)
        Xs = []
        for i in range(data.shape[0] - forecast_samples):
            Xs.append(data.iloc[i:i + forecast_samples].values)
        return np.array(Xs)

    def init_data(self, data):
        data = AnomalyDetectionModel.init_data(data)

        val_hours = int(data.shape[0] * 0.3 / 6)
        train_df_raw, val_df_raw = DataHelper.split_train_test(data, val_hours)
        val_df_raw, test_df_raw = DataHelper.split_train_test(val_df_raw, int(self.forecast_period_hours * 2))

        train_data = LstmAE.prepare_data_lstm(train_df_raw, self.forecast_period_hours)
        val_data = LstmAE.prepare_data_lstm(val_df_raw, self.forecast_period_hours)
        test_data = LstmAE.prepare_data_lstm(test_df_raw, self.forecast_period_hours)

        return train_df_raw, \
               val_df_raw, \
               test_df_raw, \
               train_data, \
               val_data, \
               test_data

    def fit(self, data):
        _, _, _, \
        train_data, \
        val_data, \
        test_data = self.init_data(data)

        timesteps = train_data.shape[1]
        num_features = train_data.shape[2]
        self.model = self.build_lstm_ae_model(timesteps, num_features)
        self.train(train_data)
        return self

    @validate_anomaly_df_schema
    def detect(self, data):
        num_features = data.shape[1]
        _, _, test_df_raw, \
        train_data, \
        val_data, \
        test_data = self.init_data(data)

        if test_data.shape[0] == 0:
            return pd.DataFrame()

        val_pred = self.predict(val_data)
        test_pred = self.predict(test_data)

        # print('test_data')
        # print(test_data)
        # print('test_pred')
        # print(test_pred)
        # print('abs_error_test')
        # print([np.abs(test_data[i] - test_pred[i]) for i in range(len(test_pred))])

        val_error_emp_covariance = self.fit_error_statistics(val_data, val_pred)

        val_distance = self.get_mahalanobis_distance(val_error_emp_covariance, val_data, val_pred)
        thresold_precentile = np.percentile(val_distance, self.threshold)
        print('thresold_precentile: {}'.format(thresold_precentile))

        test_distance = self.get_mahalanobis_distance(val_error_emp_covariance, test_data, test_pred)
        print('test_distance: {}'.format(test_distance))

        test_score_df = pd.DataFrame(test_df_raw[int(self.forecast_period_hours * DataConst.SAMPLES_PER_HOUR):])
        test_score_df['distance'] = test_distance
        test_score_df['threshold'] = thresold_precentile
        test_score_df['anomaly'] = test_score_df.distance > test_score_df.threshold
        anomalies = test_score_df[test_score_df.anomaly == True]
        anomalies = anomalies.iloc[:, :num_features]

        return anomalies

    def build_lstm_ae_model(self, timesteps, num_features):
        model = Sequential([
            LSTM(self.hidden_layer, return_sequences=True, input_shape=(timesteps, num_features), activation='tanh'),
            Dropout(self.dropout),
            LSTM(self.hidden_layer, activation='tanh'),
            Dropout(self.dropout),
            RepeatVector(timesteps),
            LSTM(self.hidden_layer, return_sequences=True, activation='tanh'),
            Dropout(self.dropout),
            LSTM(self.hidden_layer, return_sequences=True, activation='tanh'),
            Dropout(self.dropout),
            TimeDistributed(Dense(num_features))
        ])

        model.compile(loss='mse', optimizer='adam')

        return model

    def train(self, train_data):
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

        self.model.fit(
            train_data, train_data,
            epochs=100,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[es],
            shuffle=False
        )

    def predict(self, data):
        pred = self.model.predict(data)
        return pred

    @staticmethod
    def calc_mse(true, prediction):
        mse_loss = pd.DataFrame([mean_squared_error(true[i], prediction[i]) for i in range(len(prediction))], columns=['Error'])
        return mse_loss

    @staticmethod
    def fit_error_statistics(true, prediction):
        errors = np.mean([np.abs(true[i] - prediction[i]) for i in range(len(prediction))], axis=1)
        error_emp_covariance = EmpiricalCovariance().fit(errors)
        return error_emp_covariance

    @staticmethod
    def get_mahalanobis_distance(error_emp_covariance: EmpiricalCovariance(), true, prediction):
        errors = np.mean([np.abs(true[i] - prediction[i]) for i in range(len(prediction))], axis=1)
        dist = error_emp_covariance.mahalanobis(errors)
        return dist




