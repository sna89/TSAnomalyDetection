import pandas as pd
import copy


class DataHelper:
    def __init__(self):
        pass

    @staticmethod
    def _validate_data(data):
        assert isinstance(data, pd.DataFrame), "Data must be a data frame"

    @classmethod
    def pre_process(cls, df, index, pivot_column, value_columns):
        col_unique_values = df[pivot_column].unique()
        dfs = []
        for unique_value in col_unique_values:
            df_ = copy.deepcopy(df[df[pivot_column] == unique_value])
            df_ = cls._drop_duplicated_rows(df_, key_columns=[index, pivot_column])
            df_[index] = pd.to_datetime(df_[index], format="%d-%m-%y %H:%M")
            df_pivoted = df_.pivot(index=index, columns=pivot_column, values=value_columns)
            dfs.append(df_pivoted)
        return dfs

    @staticmethod
    def _drop_duplicated_rows(df_, key_columns):
        # for logging
        df_['duplicated'] = df_.duplicated(subset=key_columns)
        df_ = df_[~df_['duplicated']]
        return df_.drop(labels=['duplicated'], axis=1)
