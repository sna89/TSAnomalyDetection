import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from pylab import rcParams


class Const:
    ANOMALY_ADDITION = 2
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

        y1 = DataCreator.create_sin_wave(Const.A, Const.W, int(T / Const.DAYS))
        y2 = DataCreator.create_sin_wave(Const.A, Const.W, int(T / (Const.DAYS * Const.CYCLE_PER_DAY)))

        y1 = DataCreator.multiply_arr(y1, Const.DAYS)
        y2 = DataCreator.multiply_arr(y2, Const.DAYS * Const.CYCLE_PER_DAY)

        noise = np.random.normal(loc=0, scale=float(2 * Const.A) / 10, size=T)
        anomalies = np.asarray([Const.ANOMALY_ADDITION if i % int(T / 16) == 0
                                                          and i != 0
                                                          and i != T-1
                                else 0
                                for i in range(T)])
        anomalies_idx = np.asarray([idx if i % int(T / 16) == 0
                                                          and i != 0
                                                          and i != T-1
                                else None
                                for i, idx in enumerate(dt_index)])
        anomalies_idx = [idx for idx in anomalies_idx if idx is not None]

        y = y1 + y2 + noise + anomalies
        df = pd.DataFrame(data={'Value': y, 'index': dt_index})

        return df

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
