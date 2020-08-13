import pandas as pd
import copy
from datetime import timedelta
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass


@dataclass
class Period:
    hours: int
    days: int
    weeks: int


class DataConst:
    SAMPLES_PER_HOUR = 6


class DataHelper:
    def __init__(self):
        pass

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a data frame"

    @staticmethod
    def filter(df, index, type_column, value_column, attribute_name):
        df_ = copy.deepcopy(df[df[type_column] == attribute_name])
        df_ = DataHelper.drop_duplicated_rows(df_, key_columns=[index, value_column])
        df_.index = pd.to_datetime(df_[index], format="%d-%m-%y %H:%M", infer_datetime_format=True)
        df_ = df_[[value_column]]
        df_.rename({value_column: attribute_name}, axis=1, inplace=True)
        return df_

    @staticmethod
    def drop_duplicated_rows(df_, key_columns):
        # for logging
        df_['duplicated'] = df_.duplicated(subset=key_columns)
        df_ = df_[~df_['duplicated']]
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
    def get_first_and_last_observations(df_):
        first_obs_time = df_.index.min()
        last_obs_time = df_.index.max()
        return first_obs_time, last_obs_time

    @staticmethod
    def time_in_range(current, start, end):
        assert start <= end, 'start time must be earlier than end time'
        return (start <= current.index) & (current.index <= end)

    @staticmethod
    def scale(data, forecast_periods_hours=0):
        scaler = preprocessing.StandardScaler()

        if forecast_periods_hours > 0:
            train_df, test_df = DataHelper.split_train_test(data, forecast_periods_hours)
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
        start_time, _ = DataHelper.get_first_and_last_observations(data)
        end_time = DataHelper.get_max_idx(data, DataHelper.relative_delta_time(start_time,
                                                                    hours=period.hours,
                                                                    days=period.days,
                                                                    weeks=period.weeks))

        data_first_period = data.loc[start_time:end_time]
        return data_first_period

    @staticmethod
    def relative_delta_time(current_time, hours, days, weeks):
        return current_time + relativedelta(hours=hours) + relativedelta(days=days) + relativedelta(weeks=weeks)

    @staticmethod
    def split_train_test(data, forecast_periods_hours):
        test_periods = forecast_periods_hours * DataConst.SAMPLES_PER_HOUR
        train_periods = data.shape[0] - test_periods

        train_df = pd.DataFrame(data=data.iloc[:train_periods],
                                index=data.iloc[:train_periods].index)
        test_df = pd.DataFrame(data=data.iloc[train_periods:],
                               index=data.iloc[train_periods:].index)

        return train_df, test_df