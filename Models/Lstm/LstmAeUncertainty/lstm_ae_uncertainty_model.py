import torch.nn as nn


class LstmAeUncertaintyModel(nn.Module):
    def __init__(self, input_dim, hidden_layer, encoder_dim, dropout_p=0.2):
        super(LstmAeUncertaintyModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.encoder_dim = encoder_dim
        self.num_layers = 1

        self.encoder = nn.LSTM(self.input_dim,
                               self.encoder_dim,
                               num_layers=self.num_layers,
                               dropout=dropout_p,
                               batch_first=True)
        self.decoder = nn.LSTM(self.encoder_dim,
                               self.hidden_layer,
                               dropout=dropout_p,
                               num_layers=self.num_layers,
                               batch_first=True)
        self.fc = nn.Linear(self.hidden_layer, self.input_dim)

        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dropout_2 = nn.Dropout(p=dropout_p)

        self.activation = nn.ReLU()

    def forward(self, enc_input):
        bs = enc_input.size()[0]
        seq_len = enc_input.size()[1]

        enc_out, (hidden_enc, _) = self.encoder(enc_input)
        enc_out = self.dropout_1(self.activation(hidden_enc))
        enc_out = enc_out.view(bs, self.num_layers, self.encoder_dim) # bs, num_layers, self.encoder_dim

        dec_input = enc_out.repeat(1, seq_len, 1) # bs, seq_len * num_layers, self.encoder_dim

        dec_out, (hidden_dec, _) = self.decoder(dec_input)
        dec_out = self.dropout_2(self.activation(dec_out))  # bs, seq_len * num_layers, self.hidden_layer

        dec_out = dec_out.contiguous().view(-1, self.hidden_layer) # bs * seq_len * num_layers, self.hidden_layer

        out = self.fc(dec_out) # bs * seq_len * num_layers, self.input_dim

        out = out.view(bs, seq_len , self.input_dim)

        return out

