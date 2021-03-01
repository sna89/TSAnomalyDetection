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


class DataCreatorAnomalyMetadata:
    MAX_ANOMALY_ADDITION = 0.6
    MIN_ANOMALY_ADDITION = 0.2
    ANOMALY_DECREASE = 0.05
    ANOMALY_RATIO = 0.01
    ITERATIONS = 4


class DataCreatorHighFreqMetadata:
    HIGH_FREQ_FACTOR = 5
    NUM_HIGH_FREQ_PERIODS = 3
    HIGH_FREQ_PERIOD_DAYS = 3


class DataCreatorHolidayMetadata:
    DECREASE = 0.5


class DataCreatorWeekendMetadata:
    DECREASE = 0.5
    FRIDAY_WEEKDAY = 4
    SATURDAY_WEEKDAY = 5

class DataCreator:
    logger = get_logger("DataCreator")

    @classmethod
    def create_dataset(cls,
                       data_period,
                       train_period,
                       freq,
                       higher_freq=False,
                       is_weekends=False,
                       is_holidays=False,
                       number_of_series=1):

        start_dt = DataCreatorMetadata.START_DATE
        end_dt = DataHelper.relative_delta_time(pd.to_datetime(DataCreatorMetadata.START_DATE),
                                                minutes=data_period.minutes,
                                                hours=data_period.hours,
                                                days=data_period.days,
                                                weeks=data_period.weeks)
        cls.logger.info("Start creating synthetic dataset for {} time series: \n"
                        "start date: {},"
                        "end date: {},"
                        "freq: {}".format(number_of_series, start_dt, end_dt, freq))

        anomalies_dfs = []
        dfs = []

        dt_index = DataCreator.create_index(start_dt, end_dt, freq)
        anomaly_indices = DataCreator._get_anomaly_indices(dt_index, train_period)

        weekends_series = np.zeros(len(dt_index)) if not is_weekends else DataCreator.get_weekend_series(dt_index)
        weekends_ond_hot_df = DataCreator.get_weekend_one_hot_df(dt_index)
        dfs.append(weekends_ond_hot_df)

        holidays_series = np.zeros(len(dt_index)) if not is_holidays else DataCreator.get_holiday_series(dt_index)
        holidays_ond_hot_df = DataCreator.get_holiday_one_hot_df(dt_index)
        dfs.append(holidays_ond_hot_df)

        for series_num in range(number_of_series):
            df, anomalies_df = DataCreator.create_series(dt_index,
                                                         series_num,
                                                         anomaly_indices,
                                                         weekends_series,
                                                         holidays_series,
                                                         higher_freq)
            dfs.append(df)
            anomalies_dfs.append(anomalies_df)

        df = pd.concat(dfs, axis=1)
        df.reset_index(inplace=True)
        anomalies_df = pd.concat(anomalies_dfs, axis=1)
        anomalies_df = anomalies_df[anomalies_df.any(axis=1)]

        cls.logger.info("Synthetic data was created successfully")
        return df, anomalies_df

    @classmethod
    def create_series(cls,
                      dt_index,
                      series_num,
                      anomalies_indices,
                      weekends_series,
                      holidays_series,
                      higher_freq=False
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

        trend = DataCreator._create_trend(daily, days, daily_high_freq)
        noise = np.random.normal(loc=0, scale=float(1) / 10, size=T)
        bias = np.random.uniform(low=-0.2, high=0.2)

        anomalies = DataCreator.create_anomaly_data(T, anomalies_indices)
        anomalies_df = DataCreator.create_anomaly_df(anomalies, dt_index, series_num)

        y = trend + noise + anomalies + bias + weekends_series + holidays_series
        df = pd.DataFrame(data={'Value_{}'.format(series_num): y}, index=dt_index)

        return df, anomalies_df

    @staticmethod
    def _get_periods_in_day(dt_index):
        T = len(dt_index)
        days = DataCreator._calc_days(dt_index)
        return int(T / days)

    @staticmethod
    def _get_num_of_anomalies(dt_index, train_period):
        anomalies_start_idx = DataCreator._get_anomalies_start_idx(dt_index, train_period)
        num_anomalies = int(((len(dt_index) - anomalies_start_idx) * DataCreatorAnomalyMetadata.ANOMALY_RATIO) / 2) + 1
        return num_anomalies

    @staticmethod
    def _get_anomalies_start_idx(dt_index, train_period):
        start_dt = dt_index.min()
        anomalies_start_time = DataHelper.relative_delta_time(start_dt,
                                                              minutes=train_period.minutes,
                                                              hours=train_period.hours,
                                                              days=-train_period.days,
                                                              weeks=train_period.weeks)
        anomalies_start_idx = dt_index.slice_indexer(start=dt_index.min(), end=anomalies_start_time, step=1).stop
        return anomalies_start_idx

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

    @staticmethod
    def is_date_in_weekend(date):
        return 1 \
            if ((date.weekday() >= DataCreatorWeekendMetadata.FRIDAY_WEEKDAY) &
                (date.weekday() <= DataCreatorWeekendMetadata.SATURDAY_WEEKDAY)) \
            else 0

    @staticmethod
    def get_weekend_series(index):
        weekend_dt_index = DataCreator.get_weekend_index(index)
        weekend_series = np.where([date in weekend_dt_index
                            for date in index], -DataCreatorWeekendMetadata.DECREASE, 0)
        return weekend_series

    @staticmethod
    def get_weekend_index(index):
        weekend_dt_index = pd.DatetimeIndex(date for date in index
                                            if DataCreator.is_date_in_weekend(date))
        return weekend_dt_index

    @staticmethod
    def get_weekend_one_hot_df(dt_index):
        is_weekend_series = np.array([DataCreator.is_date_in_weekend(date) for date in dt_index])
        weekend_one_hot_df = pd.DataFrame(is_weekend_series, index=dt_index, columns=["is_weekend"])
        return weekend_one_hot_df

    @staticmethod
    def get_holiday_series(dt_index):
        holidays_dt_index = DataCreator.get_holiday_index(dt_index)
        weekend_dt_index = DataCreator.get_weekend_index(dt_index)
        only_holidays_index = pd.DatetimeIndex([date for date in dt_index
                                                if date in holidays_dt_index
                                                and date not in weekend_dt_index])
        holiday_series = np.where([date in only_holidays_index for date in dt_index],
                                  -DataCreatorHolidayMetadata.DECREASE, 0)
        return holiday_series

    @staticmethod
    def get_holiday_index(dt_index):
        us_holidays_dt_index = holidays.UnitedStates()
        holidays_dt_index = pd.DatetimeIndex(date for date in dt_index if date in us_holidays_dt_index)
        return holidays_dt_index

    @staticmethod
    def get_holiday_one_hot_df(dt_index):
        us_holidays_dt_index = holidays.UnitedStates()
        is_holiday_series = np.array([1 if date in us_holidays_dt_index else 0
                                      for date in dt_index])
        holiday_one_hot_df = pd.DataFrame(is_holiday_series, index=dt_index, columns=["is_holiday"])
        return holiday_one_hot_df

    # @classmethod
    # def output_holidays(cls, decrement_series, index):
    #     us_holidays = holidays.UnitedStates()
    #
    #     df = pd.DataFrame(data={'Value': decrement_series,
    #                             'day': index.weekday,
    #                             'holiday': [1 if date in us_holidays else 0 for date in index.date]
    #                             },
    #                       index=index)
    #     holidays_df = df[df['holiday'] == 1]
    #
    #     if not holidays_df.empty:
    #         holidays_dates = sorted(set(holidays_df.index.date))
    #         if holidays_dates:
    #             cls.logger.info("Holidays in synthetic data:")
    #             for holiday in holidays_dates:
    #                 cls.logger.info("date: {}, weekday:{}".format(holiday.strftime('%Y-%m-%d'), holiday.weekday()))
    #     else:
    #         cls.logger.info("No holidays in synthetic data.")

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
    def create_anomaly_data(T, anomalies_indices):
        anomalies = np.zeros(T)

        for anomaly_idx in anomalies_indices:
            anomaly_addition = np.random.uniform(DataCreatorAnomalyMetadata.MIN_ANOMALY_ADDITION,
                                                 DataCreatorAnomalyMetadata.MAX_ANOMALY_ADDITION)

            for iter in range(1, DataCreatorAnomalyMetadata.ITERATIONS + 1):
                curr_idx = anomaly_idx + iter - 1
                if curr_idx < T:
                    anomalies[curr_idx] = anomaly_addition - \
                                          (iter - 1) * DataCreatorAnomalyMetadata.ANOMALY_DECREASE

            if DataCreatorAnomalyMetadata.ITERATIONS > 1:
                for iter in range(1, DataCreatorAnomalyMetadata.ITERATIONS):
                    curr_idx = anomaly_idx - iter
                    anomalies[curr_idx] = anomaly_addition - \
                                          (iter * DataCreatorAnomalyMetadata.ANOMALY_DECREASE)

        return anomalies

    @staticmethod
    def create_anomaly_df(anomalies, index, label):
        anomalies_df = pd.DataFrame(data={'anomaly_{}'.format(label): anomalies},
                                    index=index)
        return anomalies_df

    @staticmethod
    def _get_anomaly_indices(dt_index, train_period):
        num_anomalies = DataCreator._get_num_of_anomalies(dt_index, train_period)
        anoamly_start_idx = DataCreator._get_anomalies_start_idx(dt_index, train_period)
        anomalies_indices = np.random.randint(low=anoamly_start_idx,
                                              high=len(dt_index) - 1,
                                              size=num_anomalies)
        return anomalies_indices