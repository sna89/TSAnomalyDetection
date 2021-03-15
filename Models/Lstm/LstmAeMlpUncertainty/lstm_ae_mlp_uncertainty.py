from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty import LstmAeUncertainty
from Models.Lstm.lstmdetector import LstmDetector
from Models.Lstm.LstmAeMlpUncertainty.lstm_ae_mlp_uncertainty_model import LstmAeMlpUncertaintyModel
import torch


LSTM_AR_MLP_UNCERTAINTY_HYPERPARAMETERS = ['batch_size',
                                           'hidden_dim',
                                           'dropout',
                                           'forecast_period',
                                           'val_ratio',
                                           'lr',
                                           'input_timesteps_period',
                                           'mlp_layers',
                                           'bootstrap',
                                           'percentile_value',
                                           'epochs',
                                           'early_stop']


class LstmAeMlpUncertainty(LstmAeUncertainty):
    def __init__(self, model_hyperparameters):
        super(LstmAeMlpUncertainty, self).__init__(model_hyperparameters)

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_AR_MLP_UNCERTAINTY_HYPERPARAMETERS,
                                                           model_hyperparameters)

        self.mlp_layers = model_hyperparameters['mlp_layers']
        self.use_categorical_columns = True

    def get_lstm_model(self, num_features):
        model = LstmAeMlpUncertaintyModel(num_features,
                                          self.hidden_dim,
                                          self.dropout,
                                          self.lr,
                                          self.epochs,
                                          self.early_stop,
                                          self.batch_size,
                                          self.horizon,
                                          self.mlp_layers,
                                          self.num_used_categorical_columns,
                                          self.device,
                                          self.model_path)
        return model.to(self.device)

    def train(self, train_dl, val_dl):
        self.model.train_ae(train_dl, val_dl, use_categorical_columns=False)
        self.model.freeze_encoder()
        self.model.train_mlp(train_dl, val_dl)







