import pandas as pd
from sklearn.metrics import confusion_matrix
from Logger.logger import get_logger
from sklearn.metrics import classification_report

class EvalBuilder:
    def __init__(self, data, y_true_df, y_pred_df):
        self.logger = get_logger(__class__.__name__)

        self.data = data
        self.y_true_df = y_true_df
        self.y_pred_df = y_pred_df

    def build(self):
        self.anomalies_res = pd.DataFrame(index=self.data.index)
        self.anomalies_res['y_true'] = self.y_true_df
        self.anomalies_res['y_pred'] = self.y_pred_df

        self.anomalies_res = self.anomalies_res.isnull()
        self.anomalies_res = 1 - self.anomalies_res.astype(int)

        self.y_true = self.anomalies_res['y_true']
        self.y_pred = self.anomalies_res['y_pred']

    def output_confusion_matrix(self):
        conf_mat = confusion_matrix(self.y_true, self.y_pred)

        self.logger.info("Confustion matrix: ")
        self.logger.info(conf_mat)

        tn, fp, fn, tp = conf_mat.ravel()
        return tn, fp, fn, tp

    def output_classification_report(self):
        self.logger.info("classification report: ")
        self.logger.info(classification_report(self.y_true, self.y_pred))
