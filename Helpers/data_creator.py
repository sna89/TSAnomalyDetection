import numpy as np
import pandas as pd
from Logger.logger import get_logger


class DataCreatorConst:
    ANOMALY_ADDITION = 4
    ANOMALY_DECREASE = 0.5
    ITERATIONS = 5
    NUN_OF_ANOMALIES = 5
    A = 1
    W = 1
    DAYS = 14
    CYCLE_PER_DAY = 4
    START_DATE = '2020-01-01 00:00'
    END_DATE = '2020-01-15 00:00'
    FREQ = '10min'

class DataCreator:
    logger = get_logger("DataCreator")

    @classmethod
    def create_data(cls, start, end, freq):
        cls.logger.info("Start creating synthetic dataset:"
                        "start date: {},"
                        "end date: {},"
                        "freq: {}".format(start, end, freq))

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

        noise = np.random.normal(loc=0, scale=float(DataCreatorConst.A) / 10, size=T)
        anomalies = DataCreator.create_anomaly_data(T)

        y = y1 + y2 + noise + anomalies
        df = pd.DataFrame(data={'Value': y, 'index': dt_index})

        # df['Value'] = df['Value'].diff(1)
        # df = df.dropna()

        anomalies_df = DataCreator.create_anomaly_df(anomalies, dt_index)

        cls.logger.info("Synthetic data was created successfully")
        return df, anomalies_df

    @classmethod
    def save_to_csv(cls, df, csv_name):
        df.to_csv(csv_name)
        cls.logger.info("Synthetic data was saved successfully - filename: {}".format(csv_name))

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
        anomalies = np.zeros(T)
        indices = np.arange(start=int(T*0.6), stop=T-1, step=1)
        for _ in range(DataCreatorConst.NUN_OF_ANOMALIES):
            anomaly_idx = np.random.choice(indices, 1, replace=True)

            for iter in range(1, DataCreatorConst.ITERATIONS + 1):
                curr_idx = anomaly_idx + iter
                if curr_idx < T:
                    anomalies[curr_idx] = DataCreatorConst.ANOMALY_ADDITION - iter*DataCreatorConst.ANOMALY_DECREASE
                    indices = indices[indices != curr_idx]

            for iter in range(1, DataCreatorConst.ITERATIONS + 1):
                curr_idx = anomaly_idx - iter
                anomalies[curr_idx] = DataCreatorConst.ANOMALY_ADDITION - \
                                      ((DataCreatorConst.ITERATIONS - iter) * DataCreatorConst.ANOMALY_DECREASE)
                indices = indices[indices != curr_idx]

        return anomalies

    @staticmethod
    def create_anomaly_df(anomalies, index):
        anomalies_dt_indices = [index[idx] for idx, anomaly in enumerate(anomalies) if anomaly != 0]
        anomalies_df = pd.DataFrame(data={'Anomaly': [anomaly for anomaly in anomalies if anomaly != 0]},
                                    index=anomalies_dt_indices)
        return anomalies_df