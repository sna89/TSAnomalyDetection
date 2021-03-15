from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty_model import LstmAeUncertaintyModel
from Models.Lstm.lstmdetector import LstmDetector
import torch


LSTM_UNCERTAINTY_HYPERPARAMETERS = ['batch_size',
                                    'hidden_dim',
                                    'dropout',
                                    'forecast_period',
                                    'val_ratio',
                                    'lr',
                                    'input_timesteps_period',
                                    'bootstrap',
                                    'percentile_value',
                                    'epochs',
                                    'early_stop']


class LstmAeUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmAeUncertainty, self).__init__(model_hyperparameters)
        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)

        self.use_categorical_columns = False

    def get_lstm_model(self, num_features):
        model = LstmAeUncertaintyModel(num_features,
                                       self.hidden_dim,
                                       self.dropout,
                                       self.lr,
                                       self.epochs,
                                       self.early_stop,
                                       self.batch_size,
                                       self.horizon,
                                       self.device,
                                       self.model_path)
        return model.to(self.device)

    def train(self, train_dl, val_dl):
        self.model.train_ae(train_dl, val_dl, self.use_categorical_columns)

    @validate_anomaly_df_schema
    def detect(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        num_features = self.get_num_features(train_df_raw.iloc[0])

        val_dataset = LstmDetector.get_tensor_dataset(x_val, y_val)
        val_dl = LstmDetector.get_dataloader(val_dataset, self.batch_size)

        test_dataset = LstmDetector.get_tensor_dataset(x_test, y_test)
        inputs, labels = test_dataset.tensors[0].type(torch.FloatTensor).to(self.device),\
                         test_dataset.tensors[1].type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        inherent_noise = self.get_inherent_noise(val_dl, num_features, use_hidden=False)
        mc_mean, mc_var, uncertainty, lower_bounds, upper_bounds = self.predict(inputs, inherent_noise, False)

        anomaly_df = self.create_anomaly_df(mc_mean,
                                            inherent_noise,
                                            mc_var,
                                            uncertainty,
                                            lower_bounds,
                                            upper_bounds,
                                            test_df_raw)
        return anomaly_df








