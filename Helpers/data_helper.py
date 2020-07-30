import pandas as pd
import copy
from datetime import timedelta
from sklearn import preprocessing


class DataHelper:
    def __init__(self):
        pass

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a data frame"

    @staticmethod
    def pivot(df, index, pivot_column, value_columns):
        col_unique_values = df[pivot_column].unique()
        processed_df = pd.DataFrame()
        for unique_value in col_unique_values:
            df_ = copy.deepcopy(df[df[pivot_column] == unique_value])
            df_ = DataHelper.drop_duplicated_rows(df_, key_columns=[index, pivot_column])
            df_[index] = pd.to_datetime(df_[index], format="%d-%m-%y %H:%M")
            df_pivoted = df_.pivot(index=index, columns=pivot_column, values=value_columns)
            processed_df = pd.concat([processed_df, df_pivoted], axis=1)
        return processed_df

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
    def scale(data):
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data[data.columns] = scaler.transform(data)
        return data