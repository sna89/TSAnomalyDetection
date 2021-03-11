from Models.anomaly_detection_model import AnomalyDetectionModel, validate_anomaly_df_schema
from Models.Lstm.LstmUncertainty.lstm_uncertainty_model import LstmUncertaintyModel
from Models.Lstm.lstmdetector import LstmDetector
import numpy as np
import torch
import torch.nn as nn
from Models.Lstm.lstmdetector import LstmDetectorConst

LSTM_UNCERTAINTY_HYPERPARAMETERS = ['hidden_dim',
                                    'batch_size',
                                    'dropout',
                                    'val_ratio',
                                    'lr',
                                    'input_timesteps_period',
                                    'forecast_period']


class LstmUncertainty(LstmDetector):
    def __init__(self, model_hyperparameters):
        super(LstmUncertainty, self).__init__(model_hyperparameters)
        AnomalyDetectionModel.validate_model_hyperpameters(LSTM_UNCERTAINTY_HYPERPARAMETERS, model_hyperparameters)

        self.use_categorical_columns = True

    def train(self, train_dl, val_dl):
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf

        early_stop_current_epochs = 0

        for i in range(LstmDetectorConst.EPOCHS):
            h = self.model.init_hidden()

            running_train_loss = 0
            self.model.train()
            for seq, labels in train_dl:
                optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                y_pred, h = self.model(seq, h)
                y_pred = y_pred.type(torch.FloatTensor).to(self.device)

                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            running_train_loss /= len(train_dl)

            num_features = self.get_num_features(seq[0])
            running_val_loss = self.get_inherent_noise(val_dl, h, num_features, use_hidden=True)

            if i % 10 == 0:
                self.logger.info(f'epoch: {i:3} train loss: {running_train_loss:10.8f} val loss: {running_val_loss:10.8f}')

            if running_val_loss <= best_val_loss:
                torch.save(self.model.state_dict(), self.model_path)
                best_val_loss = running_val_loss
                early_stop_current_epochs = 0

            else:
                early_stop_current_epochs += 1

            if early_stop_current_epochs == LstmDetectorConst.EARLY_STOP_EPOCHS:
                break

        return

    @validate_anomaly_df_schema
    def detect(self, data):
        train_df_raw, val_df_raw, test_df_raw, \
        x_train, y_train, \
        x_val, y_val, \
        x_test, y_test = self.init_data(data)

        val_dataset = LstmDetector.get_tensor_dataset(x_val, y_val)
        val_dl = LstmDetector.get_dataloader(val_dataset, self.batch_size)

        test_dataset = LstmDetector.get_tensor_dataset(x_test, y_test)
        test_inputs, test_labels = test_dataset.tensors[0].type(torch.FloatTensor).to(self.device), \
                                   test_dataset.tensors[1].type(torch.FloatTensor).to(self.device)

        num_features = self.get_num_features(train_df_raw.iloc[0])
        self.model = self.get_lstm_model(num_features)
        self.model = LstmDetector.load_model(self.model, self.model_path)

        inherent_noise = self.get_inherent_noise(val_dl, num_features, use_hidden=True)
        mc_mean, lower_bounds, upper_bounds = self.predict(test_inputs, LstmDetectorConst.BOOTSTRAP, inherent_noise,
                                                           True)

        anomaly_df = self.create_anomaly_df(mc_mean[0],
                                            lower_bounds[0],
                                            upper_bounds[0],
                                            test_df_raw,
                                            test_df_raw.index,
                                            feature_names=test_df_raw.columns)
        return anomaly_df

    def get_lstm_model(self, num_features):
        model = LstmUncertaintyModel(num_features,
                                     len(self.categorical_columns),
                                     self.hidden_dim,
                                     self.batch_size,
                                     self.dropout,
                                     self.horizon,
                                     self.device)
        return model.to(self.device)
