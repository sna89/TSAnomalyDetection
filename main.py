from seasonal_esd import SeasonalESD
import pandas as pd
from data_helper import DataHelper
from data_plotter import DataPlotter

df_ref = pd.read_csv('Sensor U106748.csv')
data_helper = DataHelper()
dfs = data_helper.pre_process(df_ref, index='Time', pivot_column='Type', value_columns=['Value', 'Unit'])
data_plotter = DataPlotter()
for df in dfs:
    DataPlotter.plot_data_distribution(df['Value'])
    DataPlotter.plot_ts_data(df['Value'])



