from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.lstm_uncertainty_detector_abc import LstmUncertaintyDetectorABC, LstmModel


LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_layer',
                                    'dropout',
                                    'lr',
                                    'val_ratio',
                                    'forecast_period_hours',
                                    'timesteps_hours']


class LstmUncertaintyDetector(LstmUncertaintyDetectorABC):
    def __init__(self, model_hyperparameters):
        super(LstmUncertaintyDetector, self).__init__(model_hyperparameters)

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)

    def get_lstm_model(self, num_features):
        model = LstmModel(num_features, self.hidden_layer, self.batch_size, self.dropout, self.device)
        return model.to(self.device)

    def fit(self, data, use_hidden=True):
        return self._fit(data, use_hidden=use_hidden)

    @validate_anomaly_df_schema
    def detect(self, data):
        return self._detect(data)








