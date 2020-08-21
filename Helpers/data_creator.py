import numpy as np
import pandas as pd


class DataCreatorConst:
    ANOMALY_ADDITION = 2
    NUN_OF_ANOMALIES = 16
    A = 1
    W = 1
    DAYS = 7
    CYCLE_PER_DAY = 4


class DataCreator:
    def __init__(self):
        pass

    @staticmethod
    def create_data(start, end, freq):
        dt_index = DataCreator.create_index(start, end, freq)
        T = len(dt_index)

        y1 = DataCreator.create_sin_wave(DataCreatorConst.A,
                                         DataCreatorConst.W,
                                         int(T / DataCreatorConst.DAYS))
        y2 = DataCreator.create_sin_wave(DataCreatorConst.A,
                                         DataCreatorConst.W,
                                         int(T / (DataCreatorConst.DAYS * DataCreatorConst.CYCLE_PER_DAY)))

        y1 = DataCreator.multiply_arr(y1, DataCreatorConst.DAYS)
        y2 = DataCreator.multiply_arr(y2, DataCreatorConst.DAYS * DataCreatorConst.CYCLE_PER_DAY)

        noise = np.random.normal(loc=0, scale=float(2 * DataCreatorConst.A) / 10, size=T)
        anomalies = DataCreator.create_anomaly_data(T)

        y = y1 + y2 + noise + anomalies
        df = pd.DataFrame(data={'Value': y, 'index': dt_index})

        anomalies_df = DataCreator.create_anomaly_df(anomalies, dt_index)

        return df, anomalies_df

    @staticmethod
    def save_to_csv(df, csv_name):
        df.to_csv(csv_name)

    @staticmethod
    def create_index(start, end, freq):
        return pd.date_range(start=start, end=end, freq=freq)

    @staticmethod
    def create_sin_wave(amplitude, freq, periods):
        return amplitude * np.sin(freq * np.linspace(0, 2 * np.pi, periods))  # cycle in 1 day

    @staticmethod
    def multiply_arr(arr, mulitply):
        return np.asarray([0] + list(arr) * mulitply)

    @staticmethod
    def create_anomaly_data(T):
        anomalies = np.asarray([DataCreatorConst.ANOMALY_ADDITION
                                if i % int(T / DataCreatorConst.NUN_OF_ANOMALIES) == 0
                                     and i != 0
                                     and i != T - 1
                                else 0
                                for i in range(T)])
        return anomalies

    @staticmethod
    def create_anomaly_df(anomalies, index):
        anomalies_idx = np.asarray([idx for i, idx in enumerate(index) if anomalies[i] != 0])
        anomalies_df = pd.DataFrame(data={'Anomaly': [anomaly for anomaly in anomalies if anomaly != 0]},
                                    index=anomalies_idx)
        return anomalies_df