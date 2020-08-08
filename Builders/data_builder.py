from typing import List
import pandas as pd
from Helpers.data_helper import DataHelper, Period
from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod


@dataclass
class PreprocessDataParams:
    test: bool
    test_period: Dict
    scale: bool


@dataclass
class FileMetadata:
    filename: str
    attribute_name: str
    time_column: str


class DataBuilder:
    def __init__(self, metadata: List, preprocess_data_params):
        self.metadata = metadata
        self.preprocess_data_params = PreprocessDataParams(**preprocess_data_params)

    def build(self):
        data_builder = None

        if len(self.metadata) == 1:
            data_builder = SingleFileDataBuilder(self.metadata, self.preprocess_data_params)

        if len(self.metadata) > 1:
            data_builder = MultipleFilesDataBuilder(self.metadata, self.preprocess_data_params)

        data = data_builder.build()
        scaler = None

        if self.preprocess_data_params.scale:
            data, scaler = DataHelper.scale(data)

        return data, scaler


class AbstractDataBuilder(ABC):
    def __init__(self, metadata, preprocess_data_params):
        self.metadata = metadata
        self.preprocess_data_params = preprocess_data_params
        self.test_period = Period(**self.preprocess_data_params.test_period)
        self.new_time_column = 'sampletime'

    @staticmethod
    def read_data(filename):
        return pd.read_csv(filename)

    def get_file_metadata(self, pos):
        return self.metadata[pos]

    @abstractmethod
    def build(self):
        return


class SingleFileDataBuilder(AbstractDataBuilder):
    def __init__(self, metadata, preprocess_data_params):
        super(SingleFileDataBuilder, self).__init__(metadata, preprocess_data_params)

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

        if self.preprocess_data_params.test:
            data = DataHelper.extract_test_period(data, self.test_period)

        return data


class MultipleFilesDataBuilder(AbstractDataBuilder):
    def __init__(self, metadata, preprocess_data_params):
        super(MultipleFilesDataBuilder, self).__init__(metadata, preprocess_data_params)

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


