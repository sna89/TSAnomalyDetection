import pandas as pd
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter


data_helper = DataHelper()
data_plotter = DataPlotter()

filename = 'Sensor U106748.csv'
df_ref = pd.read_csv(filename)
df = data_helper.pre_process(df_ref, index='Time', pivot_column='Type', value_columns=['Value', 'Unit'])
data_helper.describe(df)