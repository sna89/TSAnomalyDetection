from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.lstm_uncertainty_detector_abc import LstmUncertaintyDetectorABC, LSTM_AE_MODEL


LSTM_AE_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer',
                                       'dropout',
                                       'lr',
                                       'val_ratio',
                                       'forecast_period_hours',
                                       'timesteps_hours']


class LstmAeUncertaintyDetector(LstmUncertaintyDetectorABC):
    def __init__(self, model_hyperparameters):
        super(LstmAeUncertaintyDetector, self).__init__(model_hyperparameters)

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_AE_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)

    def get_lstm_model(self, num_features):
        model = LSTM_AE_MODEL(num_features, self.hidden_layer, self.batch_size, self.dropout, self.device)
        return model.to(self.device)

    def fit(self, data, use_hidden=True):
        return self._fit(data, use_hidden=use_hidden)

    @validate_anomaly_df_schema
    def detect(self, data):
        return self._detect(data)