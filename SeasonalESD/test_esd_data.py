from SeasonalESD.seasonal_esd import SeasonalESD
import pandas as pd
from Helpers.data_plotter import DataPlotter
from AnomalyDetectors.ad import AnomalyDetector


def test_esd_data():
    # test algorithm implementation according to
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
    data_plotter = DataPlotter()

    test_data = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01]
    test_data = pd.DataFrame(data=test_data)
    seasonal_esd_test = SeasonalESD(test_data, 0.3)
    # data_plotter.plot_data(test_data)
    task = AnomalyDetector(seasonal_esd_test)
    task.run()
    # seasonal_esd_test.run()
    # data_plotter.plot_data_distribution(test_data)
    # seasonal_esd_test.plot_residual_distribution()
    # seasonal_esd_test.plot_anomalies()

