import torch.nn as nn


class LstmUncertaintyModel(nn.Module):
    def __init__(self, num_features, num_cat_features, hidden_layer, batch_size, dropout_p, horizon, device, batch_first=True):
        super(LstmUncertaintyModel, self).__init__()

        self.num_features = num_features
        self.num_cat_features = num_cat_features
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.n_layers = 2
        self.device = device
        self.horizon = horizon

        self.dropout = nn.Dropout(dropout_p)

        self.lstm = nn.LSTM(self.num_features + self.num_cat_features,
                            self.hidden_layer,
                            dropout=dropout_p,
                            num_layers=self.n_layers,
                            batch_first=batch_first)

        self.linear = nn.Linear(self.hidden_layer, self.num_features)
        self.activation = nn.Tanh()

    def forward(self, input_seq, h, test=None):
        lstm_out, a = self.lstm(input_seq, h)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_layer)
        lstm_out = self.dropout(self.activation(lstm_out))
        predictions = self.linear(lstm_out)
        if not test:
            predictions = predictions.view(self.batch_size, -1, self.num_features)
        else:
            predictions = predictions.view(input_seq.shape[0], input_seq.shape[1], self.num_features)
        predictions = predictions[:, -self.horizon:, :]
        return predictions, h

    def init_hidden(self, batch_size=None):
        weight = next(self.parameters()).data

        if not batch_size:
            batch_size = self.batch_size

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_layer).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_layer).zero_().to(self.device))

        return hidden