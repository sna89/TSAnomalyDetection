from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.lstm_uncertainty_detector_abc import LstmUncertaintyDetectorABC, LstmAeModel


LSTM_AE_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer',
                                       'encoder_dim',
                                       'dropout',
                                       'lr',
                                       'val_ratio',
                                       'forecast_period_hours',
                                       'timesteps_hours']


class LstmAeUncertaintyDetector(LstmUncertaintyDetectorABC):
    def __init__(self, model_hyperparameters):
        super(LstmAeUncertaintyDetector, self).__init__(model_hyperparameters)

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_AE_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)
        self.encoder_dim = model_hyperparameters['encoder_dim']
        self.use_hidden = False

    def get_lstm_model(self, num_features):
        model = LstmAeModel(num_features,
                            self.hidden_layer,
                            encoder_dim=self.encoder_dim,
                            num_layers=1,
                            dropout_p=self.dropout)
        return model.to(self.device)

    def fit(self, data):
        return self._fit(data, use_hidden=self.use_hidden)

    @validate_anomaly_df_schema
    def detect(self, data):
        return self._detect(data)