from Helpers.file_helper import FileHelper
import pandas as pd
from abc import ABC, abstractmethod
from Helpers.params_helper import Metadata
from typing import Union

class DataReaderFactory:
    def __init__(self, metadata_object: dict):
        self.metadata_object = Metadata(**metadata_object)

    def get_data_reader(self):
        if self.metadata_object.source == "csv":
            return CsvDataReader()
        else:
            raise ValueError("Reader does not supported")


class DataReader(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def read_data(metadata: Union[None, dict], filename=None):
        return


class CsvDataReader(DataReader):
    def __init__(self):
        super(CsvDataReader, self).__init__()

    @staticmethod
    def read_data(metadata: Union[None, dict], filename=None) -> pd.DataFrame:
        if metadata:
            metadata = Metadata(**metadata)
            filename = metadata.filename

        filename_path = FileHelper.get_file_path(filename)
        return pd.read_csv(filename_path, dayfirst=True)