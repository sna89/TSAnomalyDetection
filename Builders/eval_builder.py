import pandas as pd
from sklearn.metrics import confusion_matrix
from Logger.logger import get_logger
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from constants import AnomalyDfColumns
import numpy as np


class EvalHelper:
    def __init__(self, prediction_df, anomalies_true_df, anomalies_pred_df):
        self.logger = get_logger(__class__.__name__)

        self.prediction_df = prediction_df
        self.eval_df = pd.DataFrame(index=prediction_df.index)
        self.y_true_df = anomalies_true_df

        feature = anomalies_pred_df[AnomalyDfColumns.Feature].unique()[0]
        self.y_pred_df = anomalies_pred_df[(anomalies_pred_df[AnomalyDfColumns.IsAnomaly] == 1) &
                                           (anomalies_pred_df[AnomalyDfColumns.Feature] == feature)]

    def build(self):
        self.eval_df['y_true'] = self.y_true_df.any(axis=1)
        self.eval_df['y_pred'] = self.y_pred_df.any(axis=1)
        self.eval_df = self.eval_df.isnull()
        self.eval_df = 1 - self.eval_df.astype(int)

    def output_confusion_matrix(self):
        conf_mat = confusion_matrix(self.eval_df['y_true'], self.eval_df['y_pred'])

        self.logger.info("Confustion matrix: ")
        self.logger.info(conf_mat)

        tn, fp, fn, tp = conf_mat.ravel()
        return tn, fp, fn, tp

    def output_classification_report(self):
        self.logger.info("classification report: ")
        self.logger.info(classification_report(self.eval_df['y_true'], self.eval_df['y_pred']))

    def output_auc(self):
        auc = roc_auc_score(self.eval_df['y_true'], self.eval_df['y_pred'])
        self.logger.info("AUC: {}".format(auc))

    def calc_coverage(self):
        feature_coverage_list =[]
        columns = self.prediction_df.columns
        features = list(self.prediction_df[AnomalyDfColumns.Feature].unique())

        if AnomalyDfColumns.LowerBound in columns and \
                AnomalyDfColumns.UpperBound in columns and \
                AnomalyDfColumns.Actual in columns:

            for feature in features:
                feature_prediction_df = self.prediction_df[AnomalyDfColumns.Feature == feature]
                feature_coverage_series = feature_prediction_df.apply(lambda row: 1 if row[AnomalyDfColumns.LowerBound] <=
                                                                     row[AnomalyDfColumns.Actual] <=
                                                                     row[AnomalyDfColumns.UpperBound]
                else 0)

                feature_coverage = feature_coverage_series.mean()
                feature_coverage_list.append(feature_coverage)
                self.logger.info("Feature {} Coverage: {}".format(feature, feature_coverage))

            avg_coverage = np.array(feature_coverage_list).mean()
            self.logger.info("Average Coverage: {}".format(avg_coverage))
