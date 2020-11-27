import numpy as np
import pandas as pd
from Logger.logger import get_logger
import holidays
from sklearn.preprocessing import StandardScaler
from Helpers.data_helper import DataHelper

class DataCreatorGeneratorConst:
    A = 1
    W = 1


class DataCreatorMetadata:
    START_DATE = '2016-01-01 08:00'
    END_DATE = '2016-02-08 08:00'
    GRANULARITY = '10min'


class DataCreatorAnomalyMetadata:
    ANOMALY_ADDITION = 2
    ANOMALY_DECREASE = 0.2
    ANOMALY_RATIO = 0.01
    ITERATIONS = 4


class DataCreatorHighFreqMetadata:
    HIGH_FREQ_FACTOR = 5
    NUM_HIGH_FREQ_PERIODS = 3
    HIGH_FREQ_PERIOD_DAYS = 3


class DataCreatorHolidayMetadata:
    DECREASE = 0.5


class DataCreator:
    logger = get_logger("DataCreator")

    @classmethod
    def create_dataset(cls,
                       start,
                       end,
                       granularity,
                       higher_freq=False,
                       weekend=False,
                       holiday=False,
                       number_of_series=1):

        cls.logger.info("Start creating synthetic dataset for {} series:"
                        "start date: {},"
                        "end date: {},"
                        "freq: {}".format(number_of_series, start, end, granularity))

        dt_index = DataCreator.create_index(start, end, granularity)
        T = len(dt_index)

        anomalies_dfs = []
        dfs = []

        if number_of_series == 1:
            shared_anomalies = np.zeros(T)
        else:
            num_anomalies = DataCreator._get_num_of_anomalies(dt_index)
            shared_anomalies = DataCreator.create_anomaly_data(T, dt_index, num_anomalies)

        for series_num in range(number_of_series):
            df, anomalies_df = DataCreator.create_series(dt_index,
                                                         series_num,
                                                         shared_anomalies,
                                                         higher_freq,
                                                         weekend,
                                                         holiday)
            dfs.append(df)
            anomalies_dfs.append(anomalies_df)

        df = pd.concat(dfs, axis=1)
        df.reset_index(inplace=True)
        anomalies_df = pd.concat(anomalies_dfs, axis=1)
        anomalies_df = anomalies_df[anomalies_df.any(axis=1)]

        # sanity check
        # anomalies_df = pd.concat([anomalies_df, shared_anomalies_df], axis=1)

        cls.logger.info("Synthetic data was created successfully")
        return df, anomalies_df

    @classmethod
    def create_series(cls,
                      dt_index,
                      series_num,
                      shared_anomalies,
                      higher_freq=False,
                      weekend=False,
                      holiday=False
                      ):
        T = len(dt_index)
        days = DataCreator._calc_days(dt_index)
        periods_in_day = DataCreator._get_periods_in_day(dt_index)

        daily = DataCreator._create_daily_seasonality(periods_in_day,
                                                      DataCreatorGeneratorConst.A,
                                                      DataCreatorGeneratorConst.W,
                                                      0,
                                                      2 * np.pi)

        daily_high_freq = np.array([])
        if higher_freq:
            daily_high_freq = DataCreator._create_daily_seasonality(periods_in_day,
                                                                    DataCreatorGeneratorConst.A,
                                                                    DataCreatorHighFreqMetadata.HIGH_FREQ_FACTOR *
                                                                    DataCreatorGeneratorConst.W,
                                                                    0,
                                                                    2 * np.pi)

        weekend_holyday_decrement = np.zeros(T)
        if weekend or holiday:
            weekend_holyday_decrement = cls._decrease_value(dt_index, weekend, holiday)
            cls.output_holidays(weekend_holyday_decrement, dt_index)

        trend = DataCreator._create_trend(daily, days, daily_high_freq)
        noise = np.random.normal(loc=0, scale=float(1) / 10, size=T)

        num_anomalies = cls._get_num_of_anomalies(dt_index)
        anomalies = DataCreator.create_anomaly_data(T, dt_index, num_anomalies)
        anomalies_with_shared = \
            np.array([shared_anomalies[i]
                      if shared_anomalies[i] > 0
                      else anomalies[i]
                      for i in range(len(anomalies))])
        anomalies_df = DataCreator.create_anomaly_df(anomalies_with_shared, dt_index, series_num)

        y = trend + weekend_holyday_decrement + noise + anomalies_with_shared
        df = pd.DataFrame(data={'Value_{}'.format(series_num): y}, index=dt_index)

        return df, anomalies_df

    @staticmethod
    def _get_periods_in_day(dt_index):
        T = len(dt_index)
        days = DataCreator._calc_days(dt_index)
        return int(T / days)

    @staticmethod
    def _get_num_of_anomalies(dt_index):
        anomalies_start_idx = DataCreator._get_anomalies_start_idx(dt_index)
        num_anomalies = int(((len(dt_index) - anomalies_start_idx) * DataCreatorAnomalyMetadata.ANOMALY_RATIO) / 2) + 1
        return num_anomalies

    @staticmethod
    def _get_anomalies_start_idx(dt_index):
        max_idx = dt_index.max()
        anomalies_start_time = DataHelper.relative_delta_time(max_idx,
                                                              minutes=0,
                                                              hours=0,
                                                              days=-7,
                                                              weeks=0)
        anomalies_start_idx = dt_index.slice_indexer(start=dt_index.min(), end=anomalies_start_time, step=1).stop
        return anomalies_start_idx

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
        jan_march_series = pd.Series(DataCreator.create_sin_wave(amplitude=DataCreatorGeneratorConst.A,
                                                                 freq=DataCreatorGeneratorConst.W,
                                                                 start=float(3)/2*np.pi,
                                                                 end=2*np.pi,
                                                                 periods=int(periods/4)))

        april_dec_series = pd.Series(DataCreator.create_sin_wave(amplitude=DataCreatorGeneratorConst.A,
                                                                 freq=DataCreatorGeneratorConst.W,
                                                                 start=0,
                                                                 end=float(3)/2*np.pi,
                                                                 periods=int(3 * periods / 4)))
        return pd.concat([jan_march_series, april_dec_series], axis=0)

    @staticmethod
    def _create_daily_seasonality(periods, amplitude, freq, start_cycle, end_cycle):
        daily_seasonality = DataCreator.create_sin_wave(amplitude=amplitude,
                                                        freq=freq,
                                                        start=start_cycle,
                                                        end=end_cycle,
                                                        periods=periods)

        return daily_seasonality

    @classmethod
    def _decrease_value(cls, index, weekend=False, holiday=False):
        decrement_dt_index = pd.DatetimeIndex([])
        weekend_dt_index = pd.DatetimeIndex([])
        holidays_dt_index = pd.DatetimeIndex([])

        if weekend:
            weekend_dt_index = pd.DatetimeIndex(date for date in index if (date.weekday() >= 4) & (date.weekday() <= 5))

        if holiday:
            us_holidays_dt_index = holidays.UnitedStates()
            holidays_dt_index = pd.DatetimeIndex(date for date in index if date in us_holidays_dt_index)

        if weekend and holiday:
            decrement_dt_index = weekend_dt_index.union(holidays_dt_index)
        elif weekend and not holiday:
            decrement_dt_index = weekend_dt_index
        elif not weekend and holiday:
            decrement_dt_index = holidays_dt_index

        decrement_series = np.where([date in decrement_dt_index
                                     for date in index],
                                    -DataCreatorHolidayMetadata.DECREASE,
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
    def _create_trend(arr, mulitplier, arr_high_freq=np.array([])):
        if arr_high_freq.size == 0:
            return DataCreator._multiply_arr(arr, mulitplier)
        else:
            return DataCreator._multiply_arr_and_combine(arr, mulitplier, arr_high_freq)

    @staticmethod
    def _multiply_arr(daily, mulitplier):
        y = list(daily) * mulitplier
        return np.asarray(y)

    @staticmethod
    def _multiply_arr_and_combine(daily, mulitplier, daily_high_freq):
        days_with_high_freq = sorted(np.random.randint(1, mulitplier, DataCreatorHighFreqMetadata.NUM_HIGH_FREQ_PERIODS))
        intersection_days = 0

        trend = np.array([])
        for i in range(DataCreatorHighFreqMetadata.NUM_HIGH_FREQ_PERIODS):
            daily_periods = 0
            if i > 0:
                daily_periods_calc = days_with_high_freq[i] - \
                                     days_with_high_freq[i - 1] - \
                                     DataCreatorHighFreqMetadata.HIGH_FREQ_PERIOD_DAYS
                if daily_periods_calc >= 0:
                    daily_periods = daily_periods_calc
                else:
                    intersection_days += daily_periods_calc
            else:
                daily_periods = days_with_high_freq[i]

            c_daily = DataCreator._multiply_arr(daily,
                                                daily_periods)
            c_daily_high_freq = DataCreator._multiply_arr(daily_high_freq,
                                                          DataCreatorHighFreqMetadata.HIGH_FREQ_PERIOD_DAYS)

            c_trend = np.concatenate([c_daily, c_daily_high_freq], axis=None)
            trend = np.concatenate([trend, c_trend], axis=None)

        c_daily = DataCreator._multiply_arr(daily, mulitplier -
                                            DataCreatorHighFreqMetadata.HIGH_FREQ_PERIOD_DAYS - \
                                            days_with_high_freq[-1] - \
                                            intersection_days)
        trend = np.concatenate([trend, c_daily], axis=None)

        return np.asarray(trend)

    @staticmethod
    def create_anomaly_data(T, dt_index, num_anomalies):
        anomalies = np.zeros(T)
        anomalies_start_idx = DataCreator._get_anomalies_start_idx(dt_index)
        indices = np.arange(start=anomalies_start_idx, stop=T-1, step=1)
        for _ in range(num_anomalies):
            anomaly_idx = np.random.choice(indices, 1, replace=False)
            anomaly_idx = anomaly_idx[0]

            for iter in range(1, DataCreatorAnomalyMetadata.ITERATIONS + 1):
                curr_idx = anomaly_idx + iter - 1
                if curr_idx < T:
                    anomalies[curr_idx] = DataCreatorAnomalyMetadata.ANOMALY_ADDITION - \
                                          (iter - 1) * DataCreatorAnomalyMetadata.ANOMALY_DECREASE
                    idx_to_remove = np.where(indices == curr_idx)
                    indices = np.delete(indices, idx_to_remove)

            if DataCreatorAnomalyMetadata.ITERATIONS > 1:
                for iter in range(1, DataCreatorAnomalyMetadata.ITERATIONS):
                    curr_idx = anomaly_idx - iter
                    anomalies[curr_idx] = DataCreatorAnomalyMetadata.ANOMALY_ADDITION - \
                                          (iter * DataCreatorAnomalyMetadata.ANOMALY_DECREASE)
                    idx_to_remove = np.where(indices == curr_idx)
                    indices = np.delete(indices, idx_to_remove)

        return anomalies

    @staticmethod
    def create_anomaly_df(anomalies, index, label):
        anomalies_df = pd.DataFrame(data={'anomaly_{}'.format(label): anomalies},
                                    index=index)
        return anomalies_df