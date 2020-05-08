import pandas as pd
from Helpers.data_helper import DataHelper
from Helpers.data_plotter import DataPlotter
from Logger.logger import create_logger
from test_esd_data import test_esd_data


create_logger()
test_esd_data()

# data_helper = DataHelper()
# data_plotter = DataPlotter()
#
# ATTRIBUTES = ['internaltemp', 'internalrh']
# filename = 'Sensor U106748.csv'
# df_ref = pd.read_csv(filename)
# df = data_helper.pre_process(df_ref, index='Time', pivot_column='Type', value_columns=['Value'])
# df.info()