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
        self.logger.info("Start reading from data sources:")
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
            self.logger.error(msg)
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
        self.logger.info("Start processing data")

        dfs = []

        data = DataHelper.drop_duplicated_rows(raw_data)
        data = self._preprocess_index(data)

        if len(self.metadata_object.attribute_names) == 1 and self.metadata_object.attribute_names[0] == 'all':
            data_col_names = [col_name for col_name in data.columns
                              if col_name != self.metadata_object.time_column
                              and 'Unnamed:' not in col_name]
            for col_name in data_col_names:
                self._add_column_to_df_list(col_name, data, dfs)
        else:
            for attribute_name in self.metadata_object.attribute_names:
                self._add_column_to_df_list(attribute_name, data, dfs)

        data = pd.concat(dfs, axis=1)
        self.logger.info("Finished processing data successfully")
        return data

    def _add_column_to_df_list(self, column_name, data, dfs):
        try:
            if column_name in data.columns:
                attribute_df = pd.DataFrame(data[column_name])
            else:
                attribute_df = DataHelper.filter(data,
                                                 type_column='Type',
                                                 value_column='Value',
                                                 attribute_name=column_name)
        except Exception as e:
            raise e

        attribute_df = self._update_schema(attribute_df, column_name)
        attribute_df = self._preprocess_attribute(attribute_df)
        dfs.append(attribute_df)

    def _update_schema(self, data, attribute_name):
        filename = self.metadata_object.filename. \
            replace('.csv', ''). \
            replace(' ', '_'). \
            lower()
        data.rename({attribute_name: attribute_name + '_' + filename},
                    axis=1,
                    inplace=True)
        data.rename_axis(self.new_time_column, inplace=True)
        return data

    def _preprocess_index(self, data):
        data.index = pd.to_datetime(data[self.metadata_object.time_column],
                                    format="%d/%m/%y %H:%M",
                                    infer_datetime_format=True)
        data.sort_index(axis=1, ascending=True, inplace=True)
        return data

    def _preprocess_attribute(self, attribute_df):
        fill_method = self.preprocess_data_params.fill

        attribute_df = self._reindex_attribute(attribute_df)
        attribute_df = self._fill_missing_data(attribute_df, method=fill_method)

        if self.preprocess_data_params.test:
            attribute_df = DataHelper.extract_first_period(attribute_df, self.test_period)

        skiprows = self.preprocess_data_params.skiprows
        if skiprows > 0:
            indices_to_drop = [i for i in range(len(attribute_df)) if i % skiprows != 0]
            attribute_df = attribute_df.drop(attribute_df.index[indices_to_drop])

        return attribute_df

    def _reindex_attribute(self, attribute_df):
        new_idx = pd.date_range(start=attribute_df.index.min(),
                                end=attribute_df.index.max(),
                                freq=self.metadata_object.freq)
        attribute_df = attribute_df.reindex(index=new_idx, copy=False)
        return attribute_df

    @staticmethod
    def _fill_missing_data(data, method='ignore'):
        if method == 'ignore':
            return data

        else:
            if method == 'bfill' or method == 'ffill':
                data.fillna(method=method, inplace=True)

            elif method == 'interpolate':
                data.interpolate(inplace=True)

            else:
                raise ValueError('No such fill method')

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
        return df

    def update_index(self, df: pd.DataFrame, start_idx: pd.DatetimeIndex, end_idx: pd.DatetimeIndex, freq='10min'):
        df, start_idx, end_idx = DataHelper.get_mutual_slice(df, start_idx, end_idx)
        df = df.reset_index()
        df[self.new_time_column] = df.apply(lambda x: DataHelper.round_to_10_minutes(x[self.new_time_column]), axis=1)
        df.set_index(keys=self.new_time_column, drop=True, inplace=True)
        new_index = DataHelper.create_new_rnd_index(start_idx, end_idx, freq)
        df = df.reindex(new_index)

        return df


