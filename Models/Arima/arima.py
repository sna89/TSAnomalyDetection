import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import pyramid as pm
from pyramid.datasets import load_wineind
import numpy as np
from Models.model import Model

class Arima(Model):
    def __init__(self, data):
        super(Arima, self).__init__(data)

    def run(self):
        stepwise_fit = pm.auto_arima(self.data, start_p=1, start_q=1,
                                     max_p=3, max_q=3, m=12,
                                     start_P=0, seasonal=True,
                                     d=1, D=1, trace=True,
                                     error_action='ignore',  # don't want to know if an order does not work
                                     suppress_warnings=True,  # don't want convergence warnings
                                     stepwise=True)  # set to stepwise
        print(stepwise_fit.summary())


if __name__ == '__main__':
    # test_data = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01]
    test_data = pd.DataFrame(load_wineind().astype(np.float64))
    test_data = pd.DataFrame(data=test_data)
    arima = Arima(test_data)
    arima.run()