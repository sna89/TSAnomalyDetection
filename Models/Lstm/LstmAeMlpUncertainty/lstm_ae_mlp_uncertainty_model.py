import torch.nn as nn
import torch
import numpy as np
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty_model import LstmAeUncertaintyModel, Encoder, Decoder


class Mlp(nn.Module):
    def __init__(self, embedding_dim, cat_features_dim, layers_dim, horizon, dropout):
        super(Mlp, self).__init__()

        self.layers = nn.ModuleList()
        input_dim = embedding_dim + cat_features_dim

        for num_layer, dim in enumerate(layers_dim):
            if num_layer == 0:
                in_dim = input_dim
            else:
                in_dim = layers_dim[num_layer - 1]

            out_dim = layers_dim[num_layer]
            self.layers.append(nn.Linear(in_dim, out_dim))
        self.layers.append(nn.Linear(layers_dim[-1], horizon))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, embedding, features):
        out = torch.cat((embedding, features), 0)
        for layer in self.layers:
            out = self.dropout(self.activation(layer(out)))
        return out


class LstmAeMlpUncertaintyModel(LstmAeUncertaintyModel):
    def __init__(self, input_size, hidden_dim, dropout, batch_size, horizon, device, mlp_layers, cat_features_dim):
        super(LstmAeMlpUncertaintyModel, self).__init__(input_size, hidden_dim, dropout, batch_size, horizon, device)

        self.mlp_layers = mlp_layers
        self.cat_features_dim = cat_features_dim
        self.mlp = Mlp(hidden_dim, cat_features_dim, mlp_layers, horizon, dropout)

    def forward(self, seq):
        #predict seq using MLP
        pass

    def get_encoder(self):
        return self.encoder

    def get_batch_seq_embedding(self, seq):
        enc_out, enc_hidden = self.encoder(seq)
        embedding = enc_hidden[0]
        return embedding

    def train_mlp(self, train_dl, val_dl, epochs, early_stop_epochs, lr, model_path):
        criterion = nn.MSELoss().to(self.device)
        mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        best_val_loss = np.inf
        early_stop_current_epochs = 0

        for epoch in range(epochs):
            running_train_loss = 0
            self.mlp.train()

            for seq, labels in train_dl:
                mlp_optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)
                seq_enc = seq[:, :, :self.input_size]
                seq_cat = seq[:, -1, self.input_size:].unsqueeze(1)

                labels = labels.type(torch.FloatTensor).to(self.device)
                outputs = torch.zeros(self.batch_size, self.horizon, self.input_size).to(self.device)

                _, encoder_hidden = self.encoder(seq_enc)
                embedding = encoder_hidden[0].view(self.batch_size, 1, -1)

                in_mlp_seq = torch.cat((seq_cat, embedding), dim=2)