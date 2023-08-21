
import torch.nn as nn
import torch
import torch.nn.functional as F
#from Decoder import *
from Encoder import *

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
    def __init__(self, hidden_size, num_layers, features_dim, hidden_layer_size, d_model, batch_first=True, enforce_info=False):
        super(SentLSTM, self).__init__()
        self.lstm = nn.LSTM(features_dim + hidden_size, hidden_size, num_layers, batch_first=batch_first)
        self.num_layers = num_layers
        self.stop_linear = nn.Linear(in_features= hidden_layer_size, out_features = 1)
        self.stop_prev_hidden = nn.Linear(in_features = hidden_size , out_features = hidden_layer_size )
        self.stop_cur_hidden  = nn.Linear( in_features = hidden_size , out_features = hidden_layer_size)
        self.fc_hidden = MLP(features_dim + d_model, hidden_layer_size, hidden_size)
        self.fc_cell = MLP(features_dim + d_model,hidden_layer_size, hidden_size)
        self.num_layers = num_layers
        #self.enforce_info = enforce_info            
    
        
        
    def init_state(self, visual_features, encoded_info):
        hidden_state = self.fc_hidden(torch.cat([torch.mean(visual_features, dim=1),torch.mean(encoded_info, dim=1)], dim=-1))
        cell_state  = self.fc_cell(torch.cat([torch.mean(visual_features, dim=1),torch.mean(encoded_info, dim=1)], dim=-1))
        #shape == (batch_size, hidden_size)
        # (D∗num_layers,batch_size, hidden_dim)
        #print("hidden", torch.transpose(hidden_state, 0, 1).unsqueeze(0).shape)
        hidden_states = hidden_state.unsqueeze(0).repeat(self.num_layers, 1,1)
        cell_states = cell_state.unsqueeze(0).repeat(self.num_layers, 1,1)
        return torch.unsqueeze(hidden_state, dim=1), (hidden_states, cell_states)

    def forward(self, context_vector, prev_hidden, prev_states):
        #model.lstm(context_vector.unsqueeze(1), prev_hidden, prev_cell_state, init=lstm_init)
        # context_vector : [batch_size, 1, visual_features_dim]
        # prev_hidden : [batch_size, 1, hidden_dim]
        #prev_hidden = torch.unsqueeze(prev_hidden, dim=1) # [batch_size, 1, hidden_dim]
        context_vector = torch.unsqueeze(context_vector, dim=1)
        
        #print("Hidden and c_vector shape: ",prev_hidden.shape, context_vector.shape, )
        #if init:
        # (hidden state, cell state).shape == (D * num_layers, batch_size(N+/-), hidden_dim) where D = 2 if bidirectional else 1
        #print(prev_states[0].shape, prev_states[1].shape, prev_hidden.shape,)
        output, (hidden_state, cell_state) = self.lstm(torch.cat([context_vector, prev_hidden], dim=-1), prev_states)
        # print("Hidden and cell state shapes : ", hidden_state.shape, cell_state.shape)
        # print("Num_layers: ", self.num_layers)
        # else:
        #     #context_vector = torch.cat([context_vector, prev_hidden])
        #     output, (hidden_state, cell_state) = self.lstm(torch.cat([context_vector, prev_hidden], dim=-1))

        stop = self.stop_linear(torch.tanh(self.stop_cur_hidden(output)+ self.stop_prev_hidden(prev_hidden)))
        #print(stop.squeeze(1).squeeze(1).shape)
        # out: [batch_size, seq_len, hidden_size]
        
        # stop shape == [batch_size]
        stop = F.sigmoid(stop.squeeze())
        
        return output, stop , (hidden_state, cell_state)


class HistoryEncoder(nn.Module):
    def __init__(self,  num_layers = 1, d_model=128, n_heads =4, dim_feedforward=256, dropout=0.1, activation=F.gelu, device=None):
        super(HistoryEncoder, self).__init__()
        encoder = EncoderLayer(d_model, n_heads,dim_feedforward , dropout=dropout,
                                                  activation=activation, batch_first=True, device=device)
        self.encoder= TextEncoder(encoder, num_layers=num_layers)
        # if d_model != visual_features_dim:
        #     self.linear = nn.Linear(d_model, visual_features_dim)
        # else:
        #     self.linear = None
        
    def forward(self, x, mask=None):
        x = self.encoder(x, src_key_padding_mask=mask)
        # if self.linear is not None:
        #     x = self.linear(x)
        return x
    
    
    
if __name__ == '__main__':
    images = torch.randn((4, 32, 256))
    
    embeddings = torch.randn(4,20,256)
    
    encoder = HistoryEncoder(num_layers = 2, d_model=256, n_heads =4, dim_feedforward=256)
  
    encoded_info = encoder(embeddings)
    #print(encoded_info.shape)
    
    #hidden_size, num_layers, features_dim, hidden_layer_size, d_model
    # sent_lstm = SentLSTM(512, 3, 256,128,256)
    
    # hidden, (init_hidden, init_cell) = sent_lstm.init_state(images, encoded_info)
    # #print(init_hidden.shape, init_cell.shape)
    # #hidden = torch.randn(4, 10)
    # #print(torch.mean(images, 1).shape, hidden.shape, init_hidden[0].shape, init_cell[0].shape)
    # hidden,  stop, (hn, cn,) = sent_lstm(torch.mean(images, 1), hidden, (init_hidden, init_cell))
    # #print(hn.shape, cn.shape,hidden.shape)
    # output2,stop, ( h, c, ) = sent_lstm(torch.mean(images, 1), hidden.squeeze(), (hn, cn))
    # output1,  stop, (hn, cn,) = sent_lstm(torch.mean(images, 1), hidden.squeeze(), (hn, cn))
    
    
    #print(stop)
    #torch.Size([3, 4, 10]) torch.Size([3, 4, 10]) torch.Size([4, 1, 10])