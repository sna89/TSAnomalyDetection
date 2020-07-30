# import pandas as pd
# import bocd
# import matplotlib.pyplot as plt
# import numpy as np
# from AnomalyDetectors.pre_process_task import PreProcessTask
# import sys
#
# if __name__ == "__main__":
#     pd.set_option('display.max_rows', None)
#     np.set_printoptions(threshold=sys.maxsize)
#
#     attribute = 'internaltemp'
#     filename = 'Sensor U106748.csv'
#
#     pre_process_task = PreProcessTask(filename)
#     data = pre_process_task.pre_process()
#
#     test_signal = np.concatenate(
#         [np.random.normal(0.7, 0.05, 300),
#          np.random.normal(1.5, 0.05, 300),
#          np.random.normal(0.6, 0.05, 300),
#          np.random.normal(1.3, 0.05, 300)])
#
#     test_data = data.iloc[:1000].values
#     # data.plot()
#     # plt.show()
#
#     bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(300), bocd.StudentT(mu=50, kappa=1, alpha=1, beta=1))
#
#     rt_mle = np.empty(test_data.shape)
#     for i, d in enumerate(test_data):
#         bc.update(d)
#         rt_mle[i] = bc.rt
#
#     plt.plot(test_data, alpha=0.5, label="observation")
#     index_changes = np.where(np.ediff1d(rt_mle)<0)[0]
#     print(index_changes)
#     plt.scatter(index_changes, test_data[index_changes], c='green', label="change point")
#     plt.show()