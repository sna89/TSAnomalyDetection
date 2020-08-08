import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from Helpers.data_plotter import DataPlotter
from AnomalyDetectors.pre_process_task import PreProcessTask
# import tensorflow as tf
pd.options.mode.chained_assignment = None


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)

    attribute = 'internaltemp'
    filename = 'Sensor U106748.csv'

    pre_process_task = PreProcessTask(filename)
    df = pre_process_task.pre_process()
    df.to_csv('sensor_u106748.csv')
    col_name = df.columns[0]
    print(col_name)
    # data_plotter = DataPlotter()
    # data_plotter.plot_ts_data(df)
    # data_plotter.plot_data_distribution(df)

    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    print("Train size: {}".format(train.shape[0]))
    print("Test size: {}".format(test.shape[0]))

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler = scaler.fit(train[[col_name]])

    train[col_name] = scaler.transform(train[[col_name]])
    test[col_name] = scaler.transform(test[[col_name]])

    time_steps = 36 # 6 hours
    X_train, y_train = create_dataset(train[[col_name]], train[[col_name]], time_steps)
    X_test, y_test = create_dataset(test[[col_name]], test[[col_name]], time_steps)

    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

    model = Sequential([
        LSTM(128, input_shape=(timesteps, num_features)),
        Dropout(0.2),
        RepeatVector(timesteps),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(num_features))
    ])

    model.compile(loss='mae', optimizer='adam')
    model.summary()