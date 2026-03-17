import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path
import numpy as np
from einops import rearrange, reduce, repeat
from layers.RevIN import RevIN


class LSTM_AE(nn.Module):
    '''
    Model with an encoder, a recurrent module, and a decoder.
    RevIN을 선택적으로 적용할 수 있습니다. (cfg.use_revin=True)
    '''

    def __init__(self, cfg):
        super(LSTM_AE, self).__init__()
        self.feature_num = cfg.dim_in
        self.batch_size = cfg.batch_size # 32
        self.rnn_type = cfg.rnn_type # LSTM
        self.rnn_inp_size = cfg.rnn_inp_size # 64
        self.rnn_hid_size = cfg.rnn_hid_size # 128
        self.nlayers = cfg.nlayers # 2
        self.dropout = cfg.dropout # 0.3
        self.res_connection = cfg.res_connection
        self.return_hiddens = cfg.return_hiddens

        # RevIN
        self.use_revin = getattr(cfg, 'use_revin', False)
        if self.use_revin:
            self.revin = RevIN(
                num_features  = self.feature_num,
                eps           = getattr(cfg, 'revin_eps', 1e-5),
                affine        = getattr(cfg, 'revin_affine', True),
                subtract_last = getattr(cfg, 'revin_subtract_last', False),
            )

        # define dropout func.
        self.drop = nn.Dropout(self.dropout)

        # encoder
        self.encoder = nn.Linear(self.feature_num, self.rnn_inp_size)

        # define RNN model
        if self.rnn_type == 'LSTM':
            self.model = nn.LSTM(input_size = self.rnn_inp_size,\
                                 hidden_size = self.rnn_hid_size,\
                                 num_layers = self.nlayers,\
                                 batch_first = True,\
                                 dropout = self.dropout)
        elif self.rnn_type == 'GRU':
            self.model = nn.GRU(self.rnn_inp_size,\
                                self.rnn_hid_size,\
                                self.nlayers,\
                                dropout = self.dropout)
        else:
            raise NotImplementedError            
        
        # decoder
        self.decoder = nn.Linear(self.rnn_hid_size, self.feature_num)

        # initialize weights
        self.init_weights()
        
    # initialize weight
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    # init hidden
    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            return torch.zeros(self.nlayers, batch_size, self.rnn_hid_size).cuda()
        elif self.rnn_type == 'LSTM':
            return (torch.zeros(self.nlayers, batch_size, self.rnn_hid_size).cuda(),
                    torch.zeros(self.nlayers, batch_size, self.rnn_hid_size).cuda())
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')
        
    def forward(self, input, input_timestamp, target, criterion, cal_score=False):
        B, S, F = input.size() # [batch_size, seq_len, feature_size]
        hidden = self.init_hidden(B)

        # ── RevIN: 정규화 ────────────────────────────────────────────────
        if self.use_revin:
            input = self.revin(input, mode='norm')
            target = self.revin(target, mode='norm')

        # LSTM embedding
        emb = self.encoder(rearrange(input, 'batch seq feature -> (batch seq) feature'))
        emb = self.drop(emb)
        emb = rearrange(emb, '(batch seq) feature -> batch seq feature', batch = B)

        # RUN LSTM
        output, hidden = self.model(emb, hidden)

        output = self.drop(output)
        decoded = self.decoder(rearrange(output, 'batch seq feature -> (batch seq) feature'))
        decoded = rearrange(decoded, '(batch seq) feature -> batch seq feature', batch = B)

        if self.res_connection:
            decoded = decoded + input

        # ── RevIN: 역정규화 ──────────────────────────────────────────────
        if self.use_revin:
            decoded = self.revin(decoded, mode='denorm')
            target  = self.revin(target,  mode='denorm')

        loss = criterion(decoded, target)
        if cal_score:
            score = self.cal_anomaly_score(decoded, input_timestamp, target, criterion)
            return decoded, loss, score
        if self.return_hiddens:
            return (decoded, hidden, output), loss
        return decoded, loss
    
    def cal_anomaly_score(self, predictions, input_timestamp, target, criterion):
        criterion = nn.MSELoss(reduction='none')
        score = criterion(predictions, target).cpu().detach().numpy()
        return np.mean(score, axis=2)