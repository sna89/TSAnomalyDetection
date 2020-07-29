from typing import List
import pandas as pd
from Helpers.data_helper import DataHelper
from dataclasses import dataclass


@dataclass
class FileMetadata:
    filename: str
    attribute_name: str
    time_column: str


class DataBuilder:
    def __init__(self, metadata: List):
        self.metadata = metadata

    def build(self):
        if len(self.metadata) == 1:
            data_builder = SingleFileDataBuilder(self.metadata)
            return data_builder.build()
        if len(self.metadata) > 1:
            data_builder = MultipleFilesDataBuilder(self.metadata)
            return data_builder.build()


class AbstractDataBuilder:
    def __init__(self, metadata):
        self.metadata = metadata
        self.new_time_column = 'sampletime'

    @staticmethod
    def read_data(filename):
        return pd.read_csv(filename)

    def get_file_metadata(self, pos):
        return self.metadata[pos]

    def build(self):
        raise NotImplementedError


class SingleFileDataBuilder(AbstractDataBuilder):
    def __init__(self, metadata):
        super().__init__(metadata)
        if isinstance(metadata, list):
            self.file_metadata = FileMetadata(**self.get_file_metadata(0))

        elif isinstance(metadata, dict):
            self.file_metadata = FileMetadata(**self.metadata)

        else:
            raise ValueError

    def build(self):
        data = SingleFileDataBuilder.read_data(self.file_metadata.filename)

        if self.file_metadata.attribute_name in data.columns:
            data.set_index(self.file_metadata.time_column, inplace=True)
            data = pd.DataFrame(data[self.file_metadata.attribute_name])
        else:
            data = DataHelper.pivot(data, index=self.file_metadata.time_column, pivot_column='Type', value_columns=['Value'])
            data = pd.DataFrame(data=data['Value'][self.file_metadata.attribute_name])

        filename = self.file_metadata.filename. \
            replace('.csv', ''). \
            replace(' ', '_'). \
            lower()
        data.rename({self.file_metadata.attribute_name: self.file_metadata.attribute_name + '_' + filename},
                    axis=1,
                    inplace=True)
        data.rename_axis(self.new_time_column, inplace=True)
        data.index = pd.to_datetime(data.index)
        return data


class MultipleFilesDataBuilder(AbstractDataBuilder):
    def __init__(self, metadata: List):
        super().__init__(metadata)

    def build(self):
        dfs = [SingleFileDataBuilder(file_metadata).build() for file_metadata in self.metadata]
        start_idx = max(list(map(lambda x: x.index.min(), dfs)))
        end_idx = min(list(map(lambda x: x.index.max(), dfs)))
        dfs = [self.update_index(df, start_idx, end_idx, '10min')
               for df in dfs]
        df = pd.concat(dfs, axis=1)
        df = df.fillna(method='bfill')
        df.to_csv('test.csv')
        return df

    def update_index(self, df: pd.DataFrame, start_idx: pd.DatetimeIndex, end_idx: pd.DatetimeIndex, freq='10min'):
        df, start_idx, end_idx = DataHelper.get_mutual_slice(df, start_idx, end_idx)
        df = df.reset_index()
        df[self.new_time_column] = df.apply(lambda x: DataHelper.round_to_10_minutes(x[self.new_time_column]), axis=1)
        df = DataHelper.drop_duplicated_rows(df, [self.new_time_column])

        df.set_index(keys=self.new_time_column, drop=True, inplace=True)
        new_index = DataHelper.create_new_rnd_index(start_idx, end_idx, freq)
        df = df.reindex(new_index)

        return df


