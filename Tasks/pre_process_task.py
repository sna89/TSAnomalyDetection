from Helpers.data_helper import DataHelper
import pandas as pd
from datetime import timedelta


class PreProcessDataTask:
    def __init__(self, *filenames, attribute='internaltemp'):
        self.filenames = filenames
        self.attribute = attribute
        self.data_helper = DataHelper()

    def pre_process(self):
        if len(self.filenames) == 1:
            filename = self.filenames[0]
            return self.pre_process_single_file(filename)
        else:
            return self.pre_process_multiple_files()

    @staticmethod
    def read_data(filename):
        return pd.read_csv(filename)

    def pre_process_single_file(self, filename):
        data = PreProcessDataTask.read_data(filename)
        data = self.data_helper.pre_process(data, index='Time', pivot_column='Type', value_columns=['Value'])
        data = pd.DataFrame(data=data['Value'][self.attribute])
        filename = filename.\
            replace('.csv', '').\
            replace(' ', '_').\
            lower()
        data = data.rename(columns={self.attribute: self.attribute + '_' + filename})
        return data

    def pre_process_multiple_files(self):
        dfs = [self.pre_process_single_file(filename) for filename in self.filenames]
        start_idx = max(list(map(lambda x: x.index.min(), dfs)))
        end_idx = min(list(map(lambda x: x.index.max(), dfs)))
        dfs = [self.update_index(df, start_idx, end_idx, '10min')
               for df in dfs]
        df = pd.concat(dfs, axis=1)
        df = df.fillna(method='bfill')
        return df

    def update_index(self, df: pd.DataFrame, start: pd.DatetimeIndex, end: pd.DatetimeIndex, freq='10min'):
        df = df.loc[start:end]
        df = df.reset_index()
        df['Time'] = df.apply(lambda x: PreProcessDataTask.round_to_10_minutes(x['Time']), axis=1)
        df = self.data_helper.drop_duplicated_rows(df, ['Time'])
        df.set_index(keys='Time', drop=True, inplace=True)
        new_index = PreProcessDataTask.create_new_rnd_index(start, end, freq)
        df = df.reindex(new_index, method='ffill')
        return df

    @staticmethod
    def create_new_rnd_index(start, end, freq):
        start = PreProcessDataTask.round_to_10_minutes(start)
        end = PreProcessDataTask.round_to_10_minutes(end)
        new_index = pd.date_range(start=start, end=end, freq=freq)
        return new_index

    @staticmethod
    def round_to_10_minutes(time):
        minute_delta = timedelta(minutes=(time.minute % 10))
        rnd_time = time - minute_delta
        return rnd_time





