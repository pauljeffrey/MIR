
import torch.nn as nn
import torch


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
    def __init__(self, input_size, hidden_size, num_layers, features_dim, enforce_info=False):
        super(SentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.stop_linear = nn.Linear(in_features= hidden_size, out_features = 1)
        self.stop_prev_hidden = nn.Linear(in_features = hidden_size , out_features = hidden_size )
        self.stop_cur_hidden  = nn.Linear( in_features = hidden_size , out_features = hidden_size)
        self.fc_hidden = MLP(features_dim, 128,hidden_size)
        self.fc_cell = MLP(features_dim, 128, hidden_size)
        self.enforce_info = enforce_info            
    
        
        
    def init_state(self, visual_features):
        hidden_state = self.fc_hidden(torch.mean(visual_features, dim=1))
        cell_state  = self.fc_cell(torch.mean(visual_features, dim=1))
        return hidden_state, cell_state

    def forward(self, context_vector, prev_hidden):
        # context_vector : [batch_size, 1, visual_features_dim]
        # prev_hidden : [batch_size, 1, hidden_dim]
        prev_hidden = torch.unsqueeze(prev_hidden, dim=1) # [batch_size, 1, hidden_dim]
        
        #context_vector = torch.cat([context_vector, prev_hidden])
        hidden, _ = self.lstm(torch.cat([context_vector, prev_hidden], dim=1))

        # out: [batch_size, seq_len, hidden_size]
        stop = nn.Sigmoid(self.stop_linear(torch.tanh(self.stop_cur_hidden(hidden)+ self.stop_prev_hidden(prev_hidden))))
        
        return hidden, stop


