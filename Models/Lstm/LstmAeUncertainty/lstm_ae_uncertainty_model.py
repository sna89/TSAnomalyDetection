# https://github.com/lkulowski/LSTM_encoder_decoder

import torch.nn as nn
import torch
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hidden_size * 2,
        #                      hidden_size=hidden_size * 2,
        #                      num_layers=num_layers,
        #                      batch_first=True)

        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x_input):
        lstm_out, hidden = self.lstm(x_input)
        lstm_out = self.dropout1(self.activation(lstm_out))

        # lstm_out, hidden = self.lstm2(lstm_out)
        # lstm_out = self.dropout2(self.activation(lstm_out))

        return lstm_out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hidden_size * 2,
        #                      hidden_size=hidden_size * 2,
        #                      num_layers=num_layers,
        #                      batch_first=True)

        self.activtion = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, hidden = self.lstm1(x_input.unsqueeze(1), encoder_hidden_states)
        lstm_out = self.dropout1(self.activtion(lstm_out))

        # lstm_out, hidden = self.lstm2(lstm_out)
        output = self.linear(lstm_out.squeeze(1))

        return output, hidden


class LstmAeUncertaintyModel(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout, batch_size, horizon, device):
        super(LstmAeUncertaintyModel, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.batch_size = batch_size
        self.horizon = horizon

        self.encoder = Encoder(input_size, hidden_dim, dropout)
        self.decoder = Decoder(input_size, hidden_dim, dropout)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def forward(self, seq):
        encoder_output, encoder_hidden = self.encoder(seq)

        outputs = torch.zeros(self.batch_size, self.horizon, seq.shape[2])

        dec_input = seq[:, -1, :]
        decoder_hidden = encoder_hidden

        for t in range(self.horizon):
            decoder_output, decoder_hidden = self.decoder(dec_input, decoder_hidden)
            outputs[:, t, :] = decoder_output
            dec_input = decoder_output

        return outputs

    def get_encoder(self):
        return self.encoder

    def get_batch_seq_embedding(self, seq):
        enc_out, enc_hidden = self.encoder(seq)
        embedding = enc_hidden[0]
        return embedding

    def train_(self, train_dl, test_dl, epochs, early_stop_epochs, lr, model_path):
        criterion = nn.MSELoss().to(self.device)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

        best_val_loss = np.inf
        early_stop_current_epochs = 0

        for epoch in range(epochs):
            running_train_loss = 0
            self.encoder.train()
            self.decoder.train()

            for seq, labels in train_dl:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                seq = seq.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)
                outputs = torch.zeros(self.batch_size, self.horizon, self.input_size).to(self.device)

                _, encoder_hidden = self.encoder(seq)
                decoder_hidden = encoder_hidden
                dec_input = seq[:, -1, :]
                # teacher forcing
                for t in range(self.horizon):
                    decoder_output, decoder_hidden = self.decoder(dec_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output
                    dec_input = labels[:, t, :]

                #Recursive
                # for t in range(self.horizon):
                #     decoder_output, decoder_hidden = self.decoder(dec_input, decoder_hidden)
                #     outputs[:, t, :] = decoder_output
                #     dec_input = decoder_output

                loss = criterion(outputs, labels)

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                running_train_loss += loss.item()

            running_train_loss /= len(train_dl)

            running_val_loss = 0
            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                for seq, labels in train_dl:
                    seq = seq.type(torch.FloatTensor).to(self.device)
                    labels = labels.type(torch.FloatTensor).to(self.device)

                    outputs = torch.zeros(self.batch_size, self.horizon, self.input_size).to(self.device)
                    _, encoder_hidden = self.encoder(seq)

                    dec_input = seq[:, -1, :]
                    decoder_hidden = encoder_hidden
                    # # teacher forcing
                    # for t in range(self.horizon):
                    #     decoder_output, decoder_hidden = self.decoder(dec_input, decoder_hidden)
                    #     outputs[:, t, :] = decoder_output
                    #     dec_input = labels[:, t, :]

                    # Recursive
                    for t in range(self.horizon):
                        decoder_output, decoder_hidden = self.decoder(dec_input, decoder_hidden)
                        outputs[:, t, :] = decoder_output
                        dec_input = decoder_output

                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            running_val_loss /= len(test_dl)

            if epoch % 10 == 0:
                print(f'epoch: {epoch:3} train loss: {running_train_loss:10.8f} val loss: {running_val_loss:10.8f}')

            if running_val_loss <= best_val_loss:
                torch.save(self.state_dict(), model_path)
                best_val_loss = running_val_loss
                early_stop_current_epochs = 0

            else:
                early_stop_current_epochs += 1

            if early_stop_current_epochs == early_stop_epochs:
                break


# class LstmAeUncertaintyModel(nn.Module):
#     def __init__(self, input_dim, hidden_layer, encoder_dim, dropout_p=0.2):
#         super(LstmAeUncertaintyModel, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_layer = hidden_layer
#         self.encoder_dim = encoder_dim
#         self.num_layers = 1
#
#         self.encoder = nn.LSTM(self.input_dim,
#                                self.encoder_dim,
#                                num_layers=self.num_layers,
#                                dropout=dropout_p,
#                                batch_first=True)
#         self.decoder = nn.LSTM(self.encoder_dim,
#                                self.hidden_layer,
#                                dropout=dropout_p,
#                                num_layers=self.num_layers,
#                                batch_first=True)
#         self.fc = nn.Linear(self.hidden_layer, self.input_dim)
#
#         self.dropout_1 = nn.Dropout(p=dropout_p)
#         self.dropout_2 = nn.Dropout(p=dropout_p)
#
#         self.activation = nn.ReLU()
#
#     def forward(self, enc_input):
#         bs = enc_input.size()[0]
#         seq_len = enc_input.size()[1]
#
#         enc_out, (hidden_enc, _) = self.encoder(enc_input)
#         enc_out = self.dropout_1(self.activation(hidden_enc))
#         enc_out = enc_out.view(bs, self.num_layers, self.encoder_dim) # bs, num_layers, self.encoder_dim
#
#         dec_input = enc_out.repeat(1, seq_len, 1) # bs, seq_len * num_layers, self.encoder_dim
#
#         dec_out, (hidden_dec, _) = self.decoder(dec_input)
#         dec_out = self.dropout_2(self.activation(dec_out))  # bs, seq_len * num_layers, self.hidden_layer
#
#         dec_out = dec_out.contiguous().view(-1, self.hidden_layer) # bs * seq_len * num_layers, self.hidden_layer
#
#         out = self.fc(dec_out) # bs * seq_len * num_layers, self.input_dim
#
#         out = out.view(bs, seq_len , self.input_dim)
#
#         return out
#
