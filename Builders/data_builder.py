from typing import List, Dict
import pandas as pd
from Helpers.data_helper import DataHelper, Period
from abc import ABC, abstractmethod
from Helpers.params_helper import Metadata, PreprocessDataParams
from Logger.logger import get_logger
from Helpers.data_reader import DataReaderFactory


class DataConstructor:
    def __init__(self, metadata: List, preprocess_data_params: Dict):
        self.metadata = metadata
        self.preprocess_data_params = PreprocessDataParams(**preprocess_data_params)
        self.logger = get_logger(__class__.__name__)

        self.raw_data = []

    def read(self):
        for metadata_object in self.metadata:
            data_reader = DataReaderFactory(metadata_object).get_data_reader()
            self.raw_data.append(data_reader.read_data(metadata_object))

        return self

    def build(self):
        if len(self.metadata) == 1:
            return SingleFileDataBuilder(self.metadata, self.preprocess_data_params)\
                .build(raw_data=self.raw_data[0])

        elif len(self.metadata) > 1:
            return MultipleFilesDataBuilder(self.metadata, self.preprocess_data_params)\
                .build(raw_data=self.raw_data)

        else:
            msg = "Metadata is empty"
            self.logger(msg)
            raise ValueError(msg)


class AbstractDataBuilder(ABC):
    def __init__(self, metadata, preprocess_data_params):
        self.metadata = metadata
        self.preprocess_data_params = preprocess_data_params

        self.test_period = Period(**self.preprocess_data_params.test_period)
        self.new_time_column = 'sampletime'

        self.logger = get_logger(__class__.__name__)

    @abstractmethod
    def build(self, raw_data):
        return


class SingleFileDataBuilder(AbstractDataBuilder):
    def __init__(self, metadata: List, preprocess_data_params: PreprocessDataParams):
        super(SingleFileDataBuilder, self).__init__(metadata, preprocess_data_params)
        self.metadata_object = Metadata(**self.metadata[0])

    def build(self, raw_data: pd.DataFrame):
        data = raw_data

        if self.metadata_object.attribute_name in data.columns:
            data.set_index(self.metadata_object.time_column, inplace=True)
            data = pd.DataFrame(data[self.metadata_object.attribute_name])
        else:
            data = DataHelper.filter(data, index=self.metadata_object.time_column,
                                     type_column='Type',
                                     value_column='Value',
                                     attribute_name=self.metadata_object.attribute_name)

        data = self.update_schema(data)
        data = self.preprocess_data(data)

        return data

    def update_schema(self, data):
        filename = self.metadata_object.filename. \
            replace('.csv', ''). \
            replace(' ', '_'). \
            lower()
        data.rename({self.metadata_object.attribute_name: self.metadata_object.attribute_name + '_' + filename},
                    axis=1,
                    inplace=True)
        data.rename_axis(self.new_time_column, inplace=True)
        data.index = pd.to_datetime(data.index)
        return data

    def preprocess_data(self, data):
        fill_method = self.preprocess_data_params.fill
        data = DataHelper.fill_missing_time(data, method=fill_method)

        if self.preprocess_data_params.test:
            data = DataHelper.extract_first_period(data, self.test_period)

        skiprows = self.preprocess_data_params.skiprows
        if skiprows > 0:
            indices_to_drop = [i for i in range(len(data)) if i % skiprows != 0]
            data = data.drop(data.index[indices_to_drop])

        return data


class MultipleFilesDataBuilder(AbstractDataBuilder):
    def __init__(self, metadata: List, preprocess_data_params: PreprocessDataParams):
        super(MultipleFilesDataBuilder, self).__init__(metadata, preprocess_data_params)

    def build(self, raw_data: List):
        dfs = [SingleFileDataBuilder(metadata_object, self.preprocess_data_params).build(raw_data[idx])
               for idx, metadata_object
               in enumerate(self.metadata)]

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


