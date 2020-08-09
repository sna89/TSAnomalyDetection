from Models.model import Model
import tensorflow as tf
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

        self.train_df, self.test_df = DataHelper.split_train_test(data, forecast_period_hours)

    @staticmethod
    def prepare_data_lstm(data, forecast_period_hours):
        forecast_samples = forecast_period_hours * DataConst.SAMPLES_PER_HOUR
        Xs = []
        for i in range(len(data) - forecast_samples):
            Xs.append(data.iloc[i:(i + forecast_samples)].values)
        return np.array(Xs)

    def run(self):
        train_data = LstmAE.prepare_data_lstm(self.train_df, self.forecast_period_hours)
        test_data = LstmAE.prepare_data_lstm(self.test_df, self.forecast_period_hours)

        timesteps = train_data.shape[1]
        num_features = train_data.shape[2]
        model = self.build_lstm_ae_model(timesteps, num_features)
        self.train(model, train_data)

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

    def train(self, model, train_data):
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

        model.fit(
            train_data, train_data,
            epochs=100,
            batch_size=16,
            validation_split=0.1,
            callbacks=[es],
            shuffle=False
        )

    def predict(self):
        pass

