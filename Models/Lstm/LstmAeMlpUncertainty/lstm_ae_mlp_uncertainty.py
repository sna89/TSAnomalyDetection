from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty import LstmAeUncertainty
from Models.Lstm.lstmdetector import LstmDetector, LstmDetectorConst
from Models.Lstm.LstmAeMlpUncertainty.lstm_ae_mlp_uncertainty_model import LstmAeMlpUncertaintyModel
import torch


LSTM_AR_MLP_UNCERTAINTY_HYPERPARAMETERS = ['batch_size',
                                           'hidden_dim',
                                           'dropout',
                                           'forecast_period',
                                           'val_ratio',
                                           'lr',
                                           'input_timesteps_period',
                                           'mlp_layers']


class LstmAeMlpUncertainty(LstmAeUncertainty):
    def __init__(self, model_hyperparameters):
        super(LstmAeMlpUncertainty, self).__init__(model_hyperparameters)

        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_AR_MLP_UNCERTAINTY_HYPERPARAMETERS,
                                                           model_hyperparameters)

        self.mlp_layers = model_hyperparameters['mlp_layers']

    def get_lstm_model(self, num_features):
        model = LstmAeMlpUncertaintyModel(num_features,
                                          self.hidden_dim,
                                          self.dropout,
                                          self.batch_size,
                                          self.horizon,
                                          self.device,
                                          self.mlp_layers,
                                          self.num_used_categorical_columns)
        return model.to(self.device)

    def train(self, train_dl, val_dl):
        epochs = LstmDetectorConst.EPOCHS
        early_stop_epochs = LstmDetectorConst.EARLY_STOP_EPOCHS
        lr = self.lr
        model_path = self.model_path

        self.model.train_ae(train_dl, val_dl, epochs, early_stop_epochs, lr, model_path, use_categorical_columns=False)
        self.model.freeze_encoder()
        self.model.train_mlp(train_dl, val_dl, epochs, early_stop_epochs, lr, model_path)

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
        inputs, labels = test_dataset.tensors[0].type(torch.FloatTensor).to(self.device), \
                         test_dataset.tensors[1].type(torch.FloatTensor).to(self.device)

        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)
        #need to load model with MLP

        inherent_noise = self.get_inherent_noise(val_dl, num_features, use_hidden=False)
        mc_mean, lower_bounds, upper_bounds = self.predict(inputs, LstmDetectorConst.BOOTSTRAP, inherent_noise, False)

        anomaly_df = self.create_anomaly_df(mc_mean,
                                            lower_bounds,
                                            upper_bounds,
                                            test_df_raw,
                                            test_df_raw.index,
                                            test_df_raw.columns)
        return anomaly_df








