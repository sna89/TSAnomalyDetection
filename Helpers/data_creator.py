import numpy as np
import pandas as pd
from Logger.logger import get_logger
import holidays
from sklearn.preprocessing import StandardScaler


class DataCreatorConst:
    ANOMALY_ADDITION = 1
    ANOMALY_DECREASE = 0.3
    ANOMALY_RATIO = 0.01
    ITERATIONS = 2
    A = 1
    W = 1
    CYCLE_PER_DAY = 1
    START_DATE = '2016-01-01 08:00'
    END_DATE = '2019-04-01 08:00'
    GRANULARITY = '10min'
    WEEKEND_DECREASE = 0.5


class DataCreator:
    logger = get_logger("DataCreator")

    @classmethod
    def create_data(cls, start, end, granularity):
        cls.logger.info("Start creating synthetic dataset:"
                        "start date: {},"
                        "end date: {},"
                        "freq: {}".format(start, end, granularity))

        dt_index = DataCreator.create_index(start, end, granularity)
        T = len(dt_index)
        days = DataCreator._calc_days(dt_index)
        years = DataCreator._calc_years(dt_index)
        num_anomalies = int(T * DataCreatorConst.ANOMALY_RATIO)

        y1 = DataCreator._create_daily_seasonality(int(T / days))
        y1 = DataCreator.multiply_arr(y1, days)

        weekend_holyday_decrement = DataCreator._decrease_value_during_weekends_and_holydays(dt_index)

        y2 = DataCreator._create_yearly_seasonality(periods=int(T / years))
        y2 = DataCreator.multiply_arr(y2, years)

        noise = np.random.normal(loc=0, scale=float(1) / 2, size=T)
        anomalies = DataCreator.create_anomaly_data(T, num_anomalies)

        y = 0.5 * y1 + 0.5 * y2 + weekend_holyday_decrement + noise + anomalies
        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))
        y = scaler.transform(y.reshape(-1, 1))
        df = pd.DataFrame(data={'Value': y.reshape(1, -1)[0], 'index': dt_index})

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
    def _calc_days(index):
        start = index.min()
        end = index.max()
        delta = end - start
        days = delta.days
        return days

    @staticmethod
    def _calc_years(index):
        start = index.min()
        end = index.max()
        delta = end - start
        years = int(delta.days / 365.25)
        return years

    @staticmethod
    def _create_yearly_seasonality(periods):
        jan_march_series = pd.Series(DataCreator.create_sin_wave(amplitude=DataCreatorConst.A,
                                                                 freq=DataCreatorConst.W,
                                                                 start=float(3)/2*np.pi,
                                                                 end=2*np.pi,
                                                                 periods=int(periods/4)))

        april_dec_series = pd.Series(DataCreator.create_sin_wave(amplitude=DataCreatorConst.A,
                                                                 freq=DataCreatorConst.W,
                                                                 start=0,
                                                                 end=float(3)/2*np.pi,
                                                                 periods=int(3 * periods / 4)))
        return pd.concat([jan_march_series, april_dec_series], axis=0)

    @staticmethod
    def _create_daily_seasonality(periods):
        daily_seasonality_series = DataCreator.create_sin_wave(amplitude=DataCreatorConst.A,
                                                               freq=DataCreatorConst.W,
                                                               start=0,
                                                               end=2*np.pi,
                                                               periods=periods) # daily seasonality
        return daily_seasonality_series

    @staticmethod
    def _decrease_value_during_weekends_and_holydays(index):
        us_holidays = holidays.UnitedStates()
        decrement_series = np.where([date in us_holidays or ((date.weekday() >= 4) & (date.weekday() <= 5))
                                     for date in index.date],
                                    -DataCreatorConst.WEEKEND_DECREASE,
                                    0)
        return decrement_series

    @staticmethod
    def create_index(start, end, freq):
        return pd.date_range(start=start, end=end, freq=freq)

    @staticmethod
    def create_sin_wave(amplitude, freq, start, end, periods):
        return amplitude * np.sin(freq * np.linspace(start, end, periods))  # cycle in 1 day

    @staticmethod
    def multiply_arr(arr, mulitply):
        return np.asarray([0] + list(arr) * mulitply)

    @staticmethod
    def create_anomaly_data(T, num_anomalies):
        anomalies = np.zeros(T)
        indices = np.arange(start=int(144), stop=T-1, step=1)
        for _ in range(num_anomalies):
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