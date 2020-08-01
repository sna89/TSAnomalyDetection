import pandas as pd


class Model:
    def __init__(self, data):
        self.data = data
        self.validate_data()
        self.data = data.iloc[:, 0]

        self.anomaly_df = None
        self.init = False

    def validate_data(self):
        assert isinstance(self.data, pd.DataFrame), "Data must be a pandas dataframe"

    def run(self):
        raise NotImplementedError