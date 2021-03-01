import torch.nn as nn
import torch
import numpy as np
from Models.Lstm.LstmAeUncertainty.lstm_ae_uncertainty_model import LstmAeUncertaintyModel, Encoder, Decoder


class Mlp(nn.Module):
    def __init__(self, embedding_dim, features_dim, layers_dim, horizon, dropout):
        super(Mlp, self).__init__()

        self.layers = []
        input_dim = embedding_dim + features_dim

        for num_layer, dim in enumerate(self.layers_dim):
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
    def __init__(self, input_size, hidden_dim, dropout, batch_size, horizon, device, mlp_layers):
        super(LstmAeMlpUncertaintyModel, self).__init__(input_size, hidden_dim, dropout, batch_size, horizon, device)

        self.mlp_layers = mlp_layers

    def forward(self, seq):
        #predict seq using MLP
        pass

    def get_encoder(self):
        return self.encoder

    def get_batch_seq_embedding(self, seq):
        enc_out, enc_hidden = self.encoder(seq)
        embedding = enc_hidden[0]
        return embedding

    def train(self):
        # train MLP after ae is pre-trained
        pass