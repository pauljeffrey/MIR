
import torch
import torch.nn as nn

from Decoder import *
from Attention import *
from lstm import *




class MedicalReportGenerator(nn.Module):
    def __init__(self, att_units, hidden_dim, features_dim, lstm_layers,  d_model, nhead, dim_feedforward, num_layers, vocab_size):
        super(MedicalReportGenerator, self).__init__()

        # Attention Module: Bahdanau Attention
        self.attention = BahdanauAttention(features_dim, hidden_dim, att_units)

        # LSTM Layer
        self.lstm = SentLSTM( features_dim + hidden_dim, hidden_dim, lstm_layers, features_dim)

        # Decoder: Transformer Decoder
        self.decoder = Decoder(d_model, nhead, dim_feedforward, num_layers, vocab_size)
        self.vocab_size = vocab_size

    