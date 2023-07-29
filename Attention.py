import torch
import torch.nn as nn

"""
This code defines a PyTorch module for the Bahdanau attention mechanism. The hidden_size parameter is the size of the hidden state 
in the encoder and decoder. The W1, W2, and v parameters are learnable weight matrices. The forward method takes a query tensor of 
shape [batch_size, hidden_size] and a values tensor of shape [batch_size, seq_len, hidden_size], where seq_len is the length of the 
input sequence. It computes scores between the query and each value in the input sequence using the W1, W2, and v parameters. It then 
computes attention weights using a softmax over the scores and computes a context vector as a weighted sum of the values using the 
attention weights. Finally, it returns the context vector and the attention weights.

"""


class BahdanauAttention(nn.Module):
    def __init__(self, features_dim, hidden_dim,  att_units, encoder_dim = None):
        super(BahdanauAttention, self).__init__()

        self.W1 = nn.Linear(hidden_dim, att_units, bias=False)
        self.W2 = nn.Linear(features_dim, att_units, bias=False)
        if encoder_dim is not None:
            self.W3 = nn.Linear(encoder_dim, att_units)
            self.patient_data_encoder_provided= True
        else:
            self.patient_data_encoder_provided= False
            
        self.v = nn.Linear(att_units, 1, bias=False)

    def forward(self, query, visual_features, patient_info=None):
        # query: [batch_size, hidden_size]
         # visual_features: [batch_size, channels, h*w]

        # Compute scores
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        if self.patient_data_encoder_provided and patient_info is not None:
            scores = self.v(torch.tanh(self.W1(query) + self.W2(visual_features) + self.W3(patient_info)))
        else:
            scores = self.v(torch.tanh(self.W1(query) + self.W2(visual_features)))  # [batch_size, seq_len, 1]

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]

        weighted_features = attn_weights * visual_features
        # Compute context vector
        context_vector = torch.sum(weighted_features, dim=1)  # [batch_size, hidden_size]
        

        return context_vector, attn_weights  


class CoAttention(nn.Module):
    def __init__(self, features_dim, semantic_dim, hidden_dim, att_units, encoder_dim=None):
        super(BahdanauAttention, self).__init__()
    
        self.W1_visual = nn.Linear(hidden_dim, att_units, bias=False)
        self.W2_visual = nn.Linear(features_dim, att_units, bias=False)
        self.v_visual = nn.Linear(att_units, 1, bias=False)
        
        if encoder_dim is not None:
            self.W3 = nn.Linear(encoder_dim, att_units)
            self.patient_data_encoder_provided= True
        else:
            self.patient_data_encoder_provided= False
        
        self.W1_semantic = nn.Linear(hidden_dim, att_units, bias=False)
        self.W2_semantic = nn.Linear(semantic_dim, att_units, bias=False)
        self.v_semantic = nn.Linear(att_units, 1, bias=False)

        self.W = nn.Linear(features_dim + semantic_dim, features_dim)
        
    def forward(self, query, visual_features, semantic_features, patient_info=None):
        # query: [batch_size, hidden_size]
        # visual_features: [batch_size, channels, h*w]
        # semantic_features: [batch_szie, n, dim]

        # Compute scores
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        if patient_info is not None and self.patient_data_encoder_provided:
            v_scores = self.v_visual(torch.tanh(self.W1_visual(query) + self.W2_visual(visual_features) + self.W3(patient_info)))  # [batch_size, seq_len, 1]
        else:
            v_scores = self.v_visual(torch.tanh(self.W1_visual(query) + self.W2_visual(visual_features)))  # [batch_size, seq_len, 1]
            

        # Compute attention weights
        visual_attn_weights = torch.softmax(v_scores, dim=1)  # [batch_size, seq_len, 1]
        
        s_scores = self.v_semantic(torch.tanh(self.W1_semantic(query) + self.W2_semantic(visual_features)))  # [batch_size, seq_len, 1]

        # Compute attention weights
        semantic_attn_weights = torch.softmax(s_scores, dim=1)  # [batch_size, seq_len, 1]


        # Compute context vector
        context_vector = self.W(torch.cat([torch.sum(visual_features * visual_attn_weights, dim=1), 
                                           torch.sum( semantic_features * semantic_attn_weights, dim=1)],dim=1))  # [batch_size, hidden_size]
        
        return   context_vector, visual_attn_weights, semantic_attn_weights
