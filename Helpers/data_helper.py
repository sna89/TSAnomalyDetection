import pandas as pd
import copy
import numpy as np


class DataHelper:
    def __init__(self):
        pass

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a data frame"

    @classmethod
    def pre_process(cls, df, index, pivot_column, value_columns):
        col_unique_values = df[pivot_column].unique()
        processed_df = pd.DataFrame()
        for unique_value in col_unique_values:
            df_ = copy.deepcopy(df[df[pivot_column] == unique_value])
            df_ = cls.drop_duplicated_rows(df_, key_columns=[index, pivot_column])
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

    # @classmethod
    # def describe(cls, df, attributes):
    #     for attribute in attributes:
    #         df_ = copy.deepcopy(df['Value'])
    #         df_[attribute] = df_[attribute].astype(float)
    #         print(df_.describe(include=[np.number]))
    #         del df_

    @staticmethod
    def split_data(df, freq='W'):
        assert isinstance(df.index, pd.DatetimeIndex), "Data frame index must be date time"
        start_period = df.index.min()
        end_period = df.index.max()
        periods = pd.DatetimeIndex([start_period])\
            .append(pd.date_range(start_period, end_period, freq=freq, normalize=True))\
            .append(pd.DatetimeIndex([end_period]))
        periods = [(periods[t],periods[t+1]) for t in range(len(periods)) if t < len(periods)-1]
        return list(map(lambda x: df[x[0]:x[1]], periods))

