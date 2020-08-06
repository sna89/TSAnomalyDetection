# from pytrends.request import TrendReq
# import matplotlib.pyplot as plt
#
#
# def get_google_trend(kw, timeframe):
#     pytrends = TrendReq()
#     kw_list = [kw]
#     pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')
#     df = pytrends.interest_over_time()
#     return df
#
#
# def process_trend_df(df, kw):
#     df = df.reset_index()
#     df = df.drop(columns=['isPartial'], axis=1)
#     df.rename(inplace=True, columns={'date': 'Date', kw: 'Trend'})
#     return df
#
#
# def plot_trend_df(df, kw):
#     df = process_trend_df(df, kw)
#     df.plot(x='Date', y='Trend', title=kw + ' ' + 'Trend', legend=False)
#     plt.show()