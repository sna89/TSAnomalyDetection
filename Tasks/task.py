from abc import ABC, abstractmethod
import pandas as pd
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter

from Logger.logger import create_logger
from dataclasses import dataclass



@dataclass
class ExperimentHyperParameters:
    train_period_weeks: int
    forecast_period_hours: int
    retrain_schedule_hours: int


class Task(ABC):
    def __init__(self, model):
        assert model is not None, 'Need to pass a class model'
        self.model = model

        self.data_helper = DataHelper()
        self.data_plotter = DataPlotter()
        create_logger()

        self.data = None

    def run(self):
        assert hasattr(self.model, 'run'), 'Model must implement "run" function'
        self.model.run()

    def read_data(self, filename):
        self.data = pd.read_csv(filename)

    @abstractmethod
    def pre_process_data(self):
        pass


