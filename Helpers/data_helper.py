import pandas as pd
import copy
from datetime import timedelta
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
from Logger.logger import get_logger
from time import time
import functools
from typing import List
import numpy as np
from Helpers.time_freq_converter import Period, TimeFreqConverter


class DataConst:
    SAMPLES_PER_HOUR = 6
    FILL_METHODS = ['ignore', 'bfill', 'ffill', 'interpolate']


class DataHelper:
    def __init__(self):
        pass

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a data frame"

    @staticmethod
    def split_df_by_columns(df, columns: List):
        df_without_columns, df_with_columns = df.drop(columns, axis=1), df[columns]
        return df_without_columns, df_with_columns

    @staticmethod
    def filter(df, type_column, value_column, attribute_name):
        df_ = copy.deepcopy(df[df[type_column] == attribute_name])
        df_ = df_[[value_column]]
        df_.rename({value_column: attribute_name}, axis=1, inplace=True)
        return df_

    @staticmethod
    def drop_duplicated_rows(df_):
        df_['duplicated'] = df_.duplicated(keep='first')
        df_ = df_[df_['duplicated'] == False]
        return df_.drop(labels=['duplicated'], axis=1)

    @staticmethod
    def split(df, freq='W'):
        assert isinstance(df.index, pd.DatetimeIndex), "Data frame index must be date time"
        start_period = df.index.min()
        end_period = df.index.max()
        periods = pd.DatetimeIndex([start_period])\
            .append(pd.date_range(start_period, end_period, freq=freq, normalize=True))\
            .append(pd.DatetimeIndex([end_period]))
        periods = [(periods[t],periods[t+1]) for t in range(len(periods)) if t < len(periods)-1]
        return list(map(lambda x: df[x[0]:x[1]], periods))

    @staticmethod
    def create_new_rnd_index(start, end, freq):
        start = DataHelper.round_to_10_minutes(start)
        end = DataHelper.round_to_10_minutes(end)
        new_index = pd.date_range(start=start, end=end, freq=freq)
        return new_index

    @staticmethod
    def round_to_10_minutes(time):
        minute_delta = timedelta(minutes=(time.minute % 10))
        second_delta = timedelta(seconds=(time.second % 60))
        rnd_time = time - minute_delta - second_delta
        return rnd_time

    @staticmethod
    def get_mutual_slice(df, start, end):
        start_idx = DataHelper.get_min_idx(df, start)
        end_idx = DataHelper.get_max_idx(df, end)
        return df.loc[start_idx:end_idx], start_idx, end_idx

    @staticmethod
    def get_min_idx(df, start):
        return df[df.index >= start].index.min()

    @staticmethod
    def get_max_idx(df, end):
        return df[df.index <= end].index.max()

    @staticmethod
    def time_in_range(current, start, end):
        assert start <= end, 'start time must be earlier than end time'
        return (start <= current.index) & (current.index <= end)

    @staticmethod
    def scale(data, forecast_periods=0):
        scaler = preprocessing.StandardScaler()

        if forecast_periods > 0:
            train_len = len(data) - forecast_periods

            train_df, test_df = DataHelper.split_train_test(data, train_len)
            scaler = scaler.fit(train_df)

            train_df[data.columns] = scaler.transform(train_df)
            test_df[data.columns] = scaler.transform(test_df)

            data = pd.concat([train_df, test_df], axis=0)

        else:
            scaler = scaler.fit(data)
            data = scaler.transform(data)

        return data, scaler

    @staticmethod
    def extract_first_period(data, period):
        start_time = data.index.min()
        end_time = DataHelper.get_max_idx(data, DataHelper.relative_delta_time(start_time,
                                                                    minutes=0,
                                                                    hours=period.hours,
                                                                    days=period.days,
                                                                    weeks=period.weeks))

        data_first_period = data.loc[start_time:end_time]
        return data_first_period

    @staticmethod
    def relative_delta_time(current_time, minutes, hours, days, weeks):
        return current_time + \
               relativedelta(minutes=minutes) + \
               relativedelta(hours=hours) + \
               relativedelta(days=days) + \
               relativedelta(weeks=weeks)

    @staticmethod
    def split_train_test(data, train_len):
        train_df = pd.DataFrame(data=data.iloc[:train_len],
                                index=data.iloc[:train_len].index)

        test_df = pd.DataFrame(data=data.iloc[train_len:],
                               index=data.iloc[train_len:].index)
        return train_df, test_df

    @staticmethod
    def is_constant_data(data):
        unique_values = data.nunique().values[0]
        return True if unique_values == 1 else False

    @staticmethod
    def interpolate(period: Period, start_val: float, end_val: float):
        periods = TimeFreqConverter.convert_to_num_samples(period, "10min")
        x = np.linspace(1, periods - 1, periods - 1)
        xp = [0, periods]
        fp = [start_val, end_val]
        interp = np.interp(x, xp, fp)
        return interp


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("timer")

        start = time()
        out = func(*args, **kwargs)
        end = time()

        logger.info("Total runtime of anomaly detection experiment: {0} minutes"
                    .format((end - start) / float(60)))
        return out
    return wrapper

