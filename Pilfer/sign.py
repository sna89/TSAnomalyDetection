import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys


class Sign:
    def __init__(self, data, alpha):
        self.data = data
        self.alpha = alpha

    def compute_scores(self, row):
        m = row.shape[0]
        scores = {}
        for key_1, value_1 in row.items():
            score=0
            for key_2, value_2 in row.items():
                if key_1 != key_2:
                    score += self.compute_score(value_1, value_2)
            score /= m-1
            scores[key_1 + '_scores'] = score
        return scores

    def compute_score(self, value_1, value_2):
        score = (value_1 - value_2)/np.absolute(value_1 - value_2)
        return score

    def get_lambda(self, v_m, v_hat):
        """
        return max(0,norm(v_m, v_hat)
        """
        return max(0, np.linalg.norm(v_m) - v_hat)

    def get_p_value(self, M, T, lamda):
        """
        return (M+1(exp(-(T*M*lambda**2)/(2*(sqrt(M)+2)**2))

        """
        return (M + 1) * np.exp(-1*((T * M * (lamda**2)) / (2 * (np.sqrt(M) + 2)**2)))

    def run(self):
        scores_df = pd.DataFrame(data=self.data.apply(lambda x: self.compute_scores(x), axis=1, result_type='expand'))
        v_m = scores_df.mean()
        v_hat = v_m.mean()
        M = v_m.shape[0]
        T = scores_df.shape[0]

        anomalies = {}

        for machine in range(M):
            lamda = self.get_lambda(v_m[machine], v_hat)
            p_value = self.get_p_value(M, T, lamda)
            if p_value <= self.alpha:
                anomalies[self.data.columns[machine]] = 1
            else:
                anomalies[self.data.columns[machine]] = 0

        anomalies_df = pd.DataFrame(data=anomalies, index=[scores_df.index.max()])
        return anomalies_df


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    np.set_printoptions(threshold=sys.maxsize)

    data = pd.read_csv('..\\test_sign.csv')
    data.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
    data.index = data.iloc[:, 0]
    data = data.iloc[:, 1:]

    # print(data.head(10))
    # print(data.columns)

    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data[data.columns] = scaler.transform(data)

    # print(scaler.mean_)
    # print(scaler.var_)
    # print(data.head(10))

    sign = Sign(data, alpha=0.5)
    sign.run()