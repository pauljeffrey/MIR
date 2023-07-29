
import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    

class SentLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, features_dim, hidden_layer_size, batch_first=True, enforce_info=False):
        super(SentLSTM, self).__init__()
        self.lstm = nn.LSTM(features_dim + hidden_size, hidden_size, num_layers, batch_first=batch_first)
        self.num_layers = num_layers
        self.stop_linear = nn.Linear(in_features= hidden_layer_size, out_features = 1)
        self.stop_prev_hidden = nn.Linear(in_features = hidden_size , out_features = hidden_layer_size )
        self.stop_cur_hidden  = nn.Linear( in_features = hidden_size , out_features = hidden_layer_size)
        self.fc_hidden = MLP(features_dim, hidden_layer_size, hidden_size)
        self.fc_cell = MLP(features_dim,hidden_layer_size, hidden_size)
        self.enforce_info = enforce_info            
    
        
        
    def init_state(self, visual_features):
        hidden_state = self.fc_hidden(torch.mean(visual_features, dim=1))
        cell_state  = self.fc_cell(torch.mean(visual_features, dim=1))
        #shape == (batch_size, output_size)
        return torch.unsqueeze(hidden_state,0), torch.unsqueeze(cell_state, 0)

    def forward(self, context_vector, prev_hidden, prev_cell_state, init):
        #model.lstm(context_vector.unsqueeze(1), prev_hidden, prev_cell_state, init=lstm_init)
        # context_vector : [batch_size, 1, visual_features_dim]
        # prev_hidden : [batch_size, 1, hidden_dim]
        prev_hidden = torch.unsqueeze(prev_hidden, dim=1) # [batch_size, 1, hidden_dim]
        
        if init:
            # (hidden state, cell state).shape == (D * num_layers, (N+/-), hidden_dim) where D = 2 if bidirectional else 1
            hidden, _ = self.lstm(torch.cat([context_vector, prev_hidden], dim=1), (prev_hidden, prev_cell_state))
        else:
            #context_vector = torch.cat([context_vector, prev_hidden])
            hidden, _ = self.lstm(torch.cat([context_vector, prev_hidden], dim=1))

        # out: [batch_size, seq_len, hidden_size]
        stop = nn.Sigmoid(self.stop_linear(torch.tanh(self.stop_cur_hidden(hidden)+ self.stop_prev_hidden(prev_hidden))))
        
        return hidden, stop


class HistoryEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers = 1, d_model=128, n_heads =4, dim_feedforward=256, dropout=0.1, activation=F.gelu, device=None):
        super._init__(self)
        self.emb_layer = nn.Embedding(vocab_size, d_model)
        encoder = nn.TransformerEncoder(d_model, n_heads,dim_feedforward , dropout=dropout,
                                                  activation=activation, batch_first=True, device=device)
        self.encoder= nn.TransformerEncoder(encoder, num_layers=num_layers)
        
    def forward(self, x):
        x = self.emb_layer(x)
        x = self.encoder(x)
        return torch.mean(x, dim=-1)