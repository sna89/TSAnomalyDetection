import numpy as np
import pandas as pd
from Logger.logger import get_logger
import holidays
from sklearn.preprocessing import StandardScaler


class DataCreatorConst:
    ANOMALY_ADDITION = 1
    ANOMALY_DECREASE = 0.3
    ANOMALY_RATIO = 0.01
    ITERATIONS = 1
    A = 1
    W = 1
    CYCLE_PER_DAY = 1
    START_DATE = '2016-01-01 08:00'
    END_DATE = '2016-03-01 08:00'
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
        # years = DataCreator._calc_years(dt_index)
        num_anomalies = int(T * DataCreatorConst.ANOMALY_RATIO)

        y1, y1_higher_freq = DataCreator._create_daily_seasonality(int(T / days), True)
        y1 = DataCreator.multiply_arr(y1, days, y1_higher_freq)

        weekend_holyday_decrement = cls._decrease_value_during_weekends_and_holydays(dt_index)
        cls.output_holidays(weekend_holyday_decrement, dt_index)

        # y2 = DataCreator._create_yearly_seasonality(periods=int(T / years))
        # y2 = DataCreator.multiply_arr(y2, years)

        noise = np.random.normal(loc=0, scale=float(1) / 5, size=T)
        anomalies = DataCreator.create_anomaly_data(T, num_anomalies, int(T/days))

        # y = 0.5 * y1 + 0.5 * y2 + weekend_holyday_decrement + noise + anomalies
        y = y1 + weekend_holyday_decrement + noise + anomalies

        df = pd.DataFrame(data={'Value': y, 'index': dt_index})

        # df['Value'] = df['Value'].diff(1)
        # df = df.dropna()

        anomalies_df = DataCreator.create_anomaly_df(anomalies, dt_index)

        cls.logger.info("Synthetic data was created successfully")
        return df, anomalies_df

    @staticmethod
    def scale(y):
        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))
        y = scaler.transform(y.reshape(-1, 1))
        y = y.reshape(1, -1)[0]
        return y

    @classmethod
    def save_to_csv(cls, df, csv_name):
        df.to_csv(csv_name)
        cls.logger.info("Synthetic data was saved successfully - filename: {}".format(csv_name))

    @staticmethod
    def _calc_days(index):
        start = index.min()
        end = index.max()
        delta = end - start
        days = delta.days + 1
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
    def _create_daily_seasonality(periods, with_faster_freq=False):
        daily_seasonality_series = DataCreator.create_sin_wave(amplitude=DataCreatorConst.A,
                                                               freq=DataCreatorConst.W,
                                                               start=0,
                                                               end=2*np.pi,
                                                               periods=periods)
        daily_seasonality_faster_freq_series = None
        if with_faster_freq:
            daily_seasonality_faster_freq_series = DataCreator.create_sin_wave(amplitude=DataCreatorConst.A,
                                                                               freq=5 * DataCreatorConst.W,
                                                                               start=0,
                                                                               end=2 * np.pi,
                                                                               periods=periods)
        return daily_seasonality_series, daily_seasonality_faster_freq_series

    @classmethod
    def _decrease_value_during_weekends_and_holydays(cls, index):
        us_holidays = holidays.UnitedStates()
        decrement_series = np.where([date in us_holidays or ((date.weekday() >= 4) & (date.weekday() <= 5))
                                     for date in index.date],
                                    -DataCreatorConst.WEEKEND_DECREASE,
                                    0)
        return decrement_series

    @classmethod
    def output_holidays(cls, decrement_series, index):
        us_holidays = holidays.UnitedStates()

        df = pd.DataFrame(data={'Value': decrement_series,
                                'day': index.weekday,
                                'holiday': [1 if date in us_holidays else 0 for date in index.date]
                                },
                          index=index)
        holidays_df = df[df['holiday'] == 1]

        if not holidays_df.empty:
            holidays_dates = sorted(set(holidays_df.index.date))
            if holidays_dates:
                cls.logger.info("Holidays in synthetic data:")
                for holiday in holidays_dates:
                    cls.logger.info("date: {}, weekday:{}".format(holiday.strftime('%Y-%m-%d'), holiday.weekday()))
        else:
            cls.logger.info("No holidays in synthetic data.")

    @staticmethod
    def create_index(start, end, freq):
        return pd.date_range(start=start, end=end, freq=freq, closed='left')

    @staticmethod
    def create_sin_wave(amplitude, freq, start, end, periods):
        return amplitude * np.sin(freq * np.linspace(start, end, periods))  # cycle in 1 day

    @staticmethod
    def multiply_arr(arr, mulitplier, arr2):
        days_high_freq = sorted(np.random.randint(1, mulitplier, 3))
        y = list(arr) * (days_high_freq[0] - 3) + \
            list(arr2) * 3 + \
            list(arr) * (days_high_freq[1] - days_high_freq[0] - 3) + \
            list(arr2) * 3 + \
            list(arr) * (days_high_freq[2] - days_high_freq[1] - 3) + \
            list(arr2) * 3 + \
            list(arr) * (mulitplier - days_high_freq[2])
        return np.asarray(y)

    @staticmethod
    def create_anomaly_data(T, num_anomalies, periods_in_day):
        anomalies = np.zeros(T)
        indices = np.arange(start=periods_in_day, stop=T-1, step=1)
        for _ in range(num_anomalies):
            anomaly_idx = np.random.choice(indices, 1, replace=False)

            for iter in range(1, DataCreatorConst.ITERATIONS + 1):
                curr_idx = anomaly_idx + iter - 1
                if curr_idx < T:
                    anomalies[curr_idx] = DataCreatorConst.ANOMALY_ADDITION - (iter - 1) * DataCreatorConst.ANOMALY_DECREASE
                    idx_to_remove = np.where(indices == curr_idx)
                    indices = np.delete(indices, idx_to_remove)

            if DataCreatorConst.ITERATIONS > 1:
                for iter in range(1, DataCreatorConst.ITERATIONS + 1):
                    curr_idx = anomaly_idx - iter
                    anomalies[curr_idx] = DataCreatorConst.ANOMALY_ADDITION - \
                                          ((DataCreatorConst.ITERATIONS - iter) * DataCreatorConst.ANOMALY_DECREASE)
                    idx_to_remove = np.where(indices == curr_idx)
                    indices = np.delete(indices, idx_to_remove)

        return anomalies

    @staticmethod
    def create_anomaly_df(anomalies, index):
        anomalies_dt_indices = [index[idx] for idx, anomaly in enumerate(anomalies) if anomaly != 0]
        anomalies_df = pd.DataFrame(data={'Anomaly': [anomaly for anomaly in anomalies if anomaly != 0]},
                                    index=anomalies_dt_indices)
        return anomalies_df