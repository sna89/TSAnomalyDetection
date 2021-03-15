import torch.nn as nn
import torch
import numpy as np
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty_model import LstmAeUncertaintyModel, Encoder, Decoder


class Mlp(nn.Module):
    def __init__(self, embedding_dim, cat_features_dim, layers_dim, horizon, output_size, dropout):
        super(Mlp, self).__init__()

        self.horizon = horizon
        self.output_size = output_size
        self.last_layer_out_dim = output_size * horizon
        self.layers = nn.ModuleList()
        self.input_dim = embedding_dim + cat_features_dim

        for num_layer, dim in enumerate(layers_dim):
            if num_layer == 0:
                in_dim = self.input_dim
            else:
                in_dim = layers_dim[num_layer - 1]

            out_dim = layers_dim[num_layer]
            self.layers.append(nn.Linear(in_dim, out_dim))
        self.layers.append(nn.Linear(layers_dim[-1], output_size * horizon))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    @staticmethod
    def prepare_input(encoder_hidden, features, batch_size):
        embedding = encoder_hidden[0].view(batch_size, -1)
        features = features.view(batch_size, -1)
        out = torch.cat((embedding, features), dim=1)
        return out

    def is_last_layer(self, layer):
        return layer.out_features == self.last_layer_out_dim

    def forward(self, encoder_hidden, features):
        batch_size = encoder_hidden[0].shape[1]
        out = self.prepare_input(encoder_hidden, features, batch_size)
        for layer in self.layers:
            if not self.is_last_layer(layer):
                out = self.dropout(self.activation(layer(out)))
            else:
                out = layer(out)
        out = out.view(batch_size, self.horizon, self.output_size)
        return out


class LstmAeMlpUncertaintyModel(LstmAeUncertaintyModel):
    def __init__(self, input_size,
                 hidden_dim,
                 dropout,
                 lr,
                 epochs,
                 early_stop,
                 batch_size,
                 horizon,
                 mlp_layers,
                 cat_features_dim,
                 device,
                 model_path):
        super(LstmAeMlpUncertaintyModel, self).__init__(input_size,
                                                        hidden_dim,
                                                        dropout,
                                                        lr,
                                                        epochs,
                                                        early_stop,
                                                        batch_size,
                                                        horizon,
                                                        device,
                                                        model_path)

        self.mlp_layers = mlp_layers
        self.cat_features_dim = cat_features_dim
        self.mlp = Mlp(hidden_dim, cat_features_dim, mlp_layers, horizon, input_size, dropout)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_batch_seq_embedding(self, seq):
        enc_out, enc_hidden = self.encoder(seq)
        embedding = enc_hidden[0]
        return embedding

    def forward(self, seq):
        seq_enc = seq[:, :, :self.input_size]
        seq_cat = seq[:, -1, self.input_size:].unsqueeze(1)
        _, encoder_hidden = self.encoder(seq_enc)

        outputs = self.mlp(encoder_hidden, seq_cat)

        return outputs

    def train_mlp(self, train_dl, val_dl):
        criterion = nn.MSELoss().to(self.device)
        mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        best_val_loss = np.inf
        early_stop_current_epochs = 0

        for epoch in range(self.epochs):
            running_train_loss = 0
            self.mlp.train()

            for seq, labels in train_dl:
                mlp_optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                outputs = self.forward(seq)

                loss = criterion(outputs, labels)
                loss.backward()
                mlp_optimizer.step()

                running_train_loss += loss.item()

            running_train_loss /= len(train_dl)

            running_val_loss = 0
            self.mlp.eval()

            with torch.no_grad():
                for seq, labels in val_dl:
                    seq = seq.type(torch.FloatTensor).to(self.device)
                    labels = labels.type(torch.FloatTensor).to(self.device)

                    outputs = self.forward(seq)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                running_val_loss /= len(val_dl)

            if epoch % 10 == 0:
                self.logger.info(f'epoch: {epoch:3} train loss: {running_train_loss:10.8f} val loss: {running_val_loss:10.8f}')

            if running_val_loss <= best_val_loss:
                torch.save(self.state_dict(), self.model_path)
                best_val_loss = running_val_loss
                early_stop_current_epochs = 0

            else:
                early_stop_current_epochs += 1

            if early_stop_current_epochs == self.early_stop:
                break