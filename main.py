from google_trend import get_google_trend
from seasonal_esd import SeasonalESD
import pandas as pd

kw = "Michal Jordan"
timeframe='2014-01-01 2020-01-01'
df = get_google_trend(kw, timeframe)
seasonal_esd_mj = SeasonalESD(df[kw], 0.05, True)
seasonal_esd_mj.run()
# SeasonalESD.plot_data_distribution(df[kw])
# seasonal_esd_mj.plot_residual_distribution()
seasonal_esd_mj.plot()
# #
# kw = "covid-19"
# timeframe='2019-10-01 2020-05-01'
# df = get_google_trend(kw, timeframe)
# seasonal_esd_covid = SeasonalESD(df[kw], 0.001, True)
# seasonal_esd_covid.run()
# seasonal_esd_covid.plot()
#
# kw = "sunderland"
# timeframe='2014-01-01 2020-05-01'
# df = get_google_trend(kw, timeframe)
# seasonal_esd_sunderland = SeasonalESD(df[kw], 0.1, True)
# seasonal_esd_sunderland.run()
# seasonal_esd_sunderland.plot()

# kw = "diet"
# timeframe='2014-01-01 2020-01-01'
# df = get_google_trend(kw, timeframe)
# seasonal_esd_diet = SeasonalESD(df[kw], 0.11, True)
# seasonal_esd_diet.run()
# seasonal_esd_diet.plot()

# kw = "china"
# timeframe='2014-01-01 2020-01-01'
# df = get_google_trend(kw, timeframe)
# seasonal_esd_china = SeasonalESD(df[kw], 0.05, True)
# seasonal_esd_china.run()
# # SeasonalESD.plot_data_distribution(df[kw])
# # seasonal_esd_china.plot_residual_distribution()
# seasonal_esd_china.plot()
#
# # test algorithm implementation according to
# # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
# test_data = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01]
# test_data = pd.DataFrame(data=test_data)
# seasonal_esd_test = SeasonalESD(test_data, 0.3)
# seasonal_esd_test.run()
# SeasonalESD.plot_data_distribution(test_data)
# seasonal_esd_test.plot_residual_distribution()
# seasonal_esd_test.plot()
