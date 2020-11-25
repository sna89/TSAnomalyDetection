import pandas as pd
from sklearn.metrics import confusion_matrix
from Logger.logger import get_logger
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


class EvalHelper:
    def __init__(self, data, y_true_df, y_pred_df):
        self.logger = get_logger(__class__.__name__)

        self.eval_df = pd.DataFrame(index=data.index)
        self.y_true_df = y_true_df
        self.y_pred_df = y_pred_df

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
        auc = roc_auc_score(self.eval_df['y_true'],self.eval_df['y_pred'])
        self.logger.info("AUC: {}".format(auc))