import pandas as pd
from Logger.logger import get_logger
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, data):
        self.data = data
        self.validate_data()
        self.data = data.iloc[:, 0]

        self.anomaly_df = None
        self.init = False

        self.logger = get_logger(__class__.__name__)

    def validate_data(self):
        assert isinstance(self.data, pd.DataFrame), "Data must be a pandas dataframe"

    @abstractmethod
    def run(self):
        pass