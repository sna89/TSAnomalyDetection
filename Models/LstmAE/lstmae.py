from Models.model import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras
import pandas as pd
import numpy as np
from Helpers.data_helper import DataHelper, DataConst

pd.options.mode.chained_assignment = None


class LstmAE(Model):
    def __init__(self, data, hidden_layer, dropout, batch_size, threshold, forecast_period_hours):
        super(LstmAE, self).__init__(data)
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.batch_size = batch_size
        self.threshold = threshold
        self.forecast_period_hours = forecast_period_hours

        self.model = None

    @staticmethod
    def prepare_data_lstm(data, forecast_period_hours):
        forecast_samples = forecast_period_hours * DataConst.SAMPLES_PER_HOUR
        Xs = []
        for i in range(len(data) - forecast_samples):
            Xs.append(data.iloc[i:(i + forecast_samples)].values)
        return np.array(Xs)

    def run(self):
        train_df_raw, test_df_raw = DataHelper.split_train_test(self.data, self.forecast_period_hours * 2)

        train_data = LstmAE.prepare_data_lstm(train_df_raw, self.forecast_period_hours)
        test_data = LstmAE.prepare_data_lstm(test_df_raw, self.forecast_period_hours)

        timesteps = train_data.shape[1]
        num_features = train_data.shape[2]
        self.model = self.build_lstm_ae_model(timesteps, num_features)
        self.train(train_data)

        train_pred = self.predict(train_data)
        test_pred = self.predict(test_data)

        train_mse_loss = self.calc_mse(train_data, train_pred)
        test_mse_loss = self.calc_mse(test_data, test_pred)

        thresold_precentile = np.percentile(train_mse_loss, self.threshold)
        test_score_df = pd.DataFrame(test_df_raw[self.forecast_period_hours * DataConst.SAMPLES_PER_HOUR:])
        test_score_df['loss'] = test_mse_loss.values
        test_score_df['threshold'] = thresold_precentile

        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        anomalies = test_score_df[test_score_df.anomaly == True]
        anomalies = anomalies.drop(columns=['loss', 'threshold', 'anomaly'])

        return anomalies

    def build_lstm_ae_model(self, timesteps, num_features):
        model = Sequential([
            LSTM(self.hidden_layer, input_shape=(timesteps, num_features), activation='tanh'),
            Dropout(self.dropout),
            RepeatVector(timesteps),
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
            batch_size=16,
            validation_split=0.1,
            callbacks=[es],
            shuffle=False
        )

    def predict(self, data):
        pred = self.model.predict(data)
        return pred

    @staticmethod
    def calc_mse(true, pred):
        mse_loss = pd.DataFrame(np.mean((pred - true) ** 2, axis=1), columns=['Error'])
        return mse_loss