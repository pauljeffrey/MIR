import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

"""
This code defines a PyTorch module for the Bahdanau attention mechanism. The hidden_size parameter is the size of the hidden state 
in the encoder and decoder. The W1, W2, and v parameters are learnable weight matrices. The forward method takes a query tensor of 
shape [batch_size, hidden_size] and a values tensor of shape [batch_size, seq_len, hidden_size], where seq_len is the length of the 
input sequence. It computes scores between the query and each value in the input sequence using the W1, W2, and v parameters. It then 
computes attention weights using a softmax over the scores and computes a context vector as a weighted sum of the values using the 
attention weights. Finally, it returns the context vector and the attention weights.

"""

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

class BahdanauAttention(nn.Module):
    def __init__(self, features_dim, hidden_dim,  att_units):
        super(BahdanauAttention, self).__init__()

        self.W1 = nn.Linear(hidden_dim, att_units, bias=False)
        self.W2 = nn.Linear(features_dim, att_units, bias=False)
       
        #self.W3 = nn.Linear(d_model, att_units)
        #self.patient_data_encoder_provided= True
        # else:
        #     self.patient_data_encoder_provided= False
            
        self.v = nn.Linear(att_units, 1, bias=False)

    def forward(self, query, visual_features): #, patient_info=None
        # query: [batch_size, hidden_size]
         # visual_features: [batch_size, channels, h*w]

        # Compute scores
        #query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        #if self.patient_data_encoder_provided and patient_info is not None:
        scores = self.v(torch.tanh(self.W1(query) + self.W2(visual_features)))
        # else:
        #     scores = self.v(torch.tanh(self.W1(query) + self.W2(visual_features)))  # [batch_size, seq_len, 1]

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]

        weighted_features = attn_weights * visual_features
        # Compute context vector
        context_vector = torch.sum(weighted_features, dim=1)  # [batch_size, hidden_size]
        

        return context_vector, attn_weights  
    

# class PromptAttention(nn.Module):
#     def __init__(self, features_dim, hidden_dim,  att_units, d_model):
#         super(BahdanauAttention, self).__init__()

#         self.W1 = nn.Linear(hidden_dim, att_units, bias=False)
#         self.W2 = nn.Linear(features_dim, att_units, bias=False)
       
#         self.W3 = nn.Linear(d_model, att_units)
#         #self.patient_data_encoder_provided= True
#         # else:
#         #     self.patient_data_encoder_provided= False
            
#         self.v = nn.Linear(att_units, 1, bias=False)

#     def forward(self, query, visual_features, patient_info=None):
#         # query: [batch_size, hidden_size]
#          # visual_features: [batch_size, channels, h*w]

#         # Compute scores
#         #query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
#         #if self.patient_data_encoder_provided and patient_info is not None:
#         scores = self.v(torch.tanh(self.W1(query) + self.W2(visual_features) +  self.W3(torch.mean(patient_info,dim=1,keepdim=True))))
#         # else:
#         #     scores = self.v(torch.tanh(self.W1(query) + self.W2(visual_features)))  # [batch_size, seq_len, 1]

#         # Compute attention weights
#         attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]

#         weighted_features = attn_weights * visual_features
#         # Compute context vector
#         context_vector = torch.sum(weighted_features, dim=1)  # [batch_size, hidden_size]
        

#         return context_vector, attn_weights  
    

class CoAttention(nn.Module):
    def __init__(self, features_dim, semantic_dim, hidden_dim, att_units, d_model):
        super(CoAttention, self).__init__()
    
        self.W1_visual = nn.Linear(hidden_dim, att_units, bias=False)
        self.W2_visual = nn.Linear(features_dim, att_units, bias=False)
        self.v_visual = nn.Linear(att_units, 1, bias=False)
        
        #if encoder_dim is not None:
        self.W3 = nn.Linear(d_model, att_units)
        self.patient_data_encoder_provided= True
        # else:
        #     self.patient_data_encoder_provided= False
        
        self.W1_semantic = nn.Linear(hidden_dim, att_units, bias=False)
        self.W2_semantic = nn.Linear(semantic_dim, att_units, bias=False)
        self.v_semantic = nn.Linear(att_units, 1, bias=False)

        self.W = nn.Linear(features_dim + semantic_dim, features_dim)
        
    def forward(self, query, visual_features, semantic_features, patient_info):
        # query: [batch_size, hidden_size]
        # visual_features: [batch_size, channels, h*w]
        # semantic_features: [batch_szie, n, dim]
        # Compute scores
        #query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        #if patient_info is not None and self.patient_data_encoder_provided:
        v_scores = self.v_visual(torch.tanh(self.W1_visual(query) + self.W2_visual(visual_features) + self.W3(torch.mean(patient_info, dim=1, keepdim=True))))  # [batch_size, seq_len, 1]
        # else:
        #     v_scores = self.v_visual(torch.tanh(self.W1_visual(query) + self.W2_visual(visual_features)))  # [batch_size, seq_len, 1]
            

        # Compute attention weights
        visual_attn_weights = torch.softmax(v_scores, dim=1)  # [batch_size, seq_len, 1]
        
        s_scores = self.v_semantic(torch.tanh(self.W1_semantic(query) + self.W2_semantic(semantic_features) + self.W3(torch.mean(patient_info, dim=1, keepdim=True))))  # [batch_size, seq_len, 1]

        # Compute attention weights
        semantic_attn_weights = torch.softmax(s_scores, dim=1)  # [batch_size, seq_len, 1]
        # print("scores shape: ", v_scores.shape, s_scores.shape)
        # print("visual results: ", (visual_attn_weights * visual_features).shape)
        # print("visual weights: ", (visual_attn_weights).shape)
        # print("semantic result: ", (semantic_attn_weights * semantic_features).shape)
        # print("semantic weight: ", (semantic_attn_weights).shape)
        # print("semantic features shape: ", semantic_features.shape)
        # print("visual features shape: ", visual_features.shape)
        # Compute context vector
        context_vector = self.W(torch.cat([torch.sum(visual_features * visual_attn_weights, dim=1), 
                                           torch.sum( semantic_features * semantic_attn_weights, dim=1)],dim=1))  # [batch_size, hidden_size]
        
        return   context_vector, visual_attn_weights, semantic_attn_weights


class MHA(nn.Module):
    def __init__(self,d_model, nhead, dropout=0.1, use_residual=True, batch_first=True, device=None, dtype=None):
        super(MHA, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.use_residual_connection = use_residual
        
    def forward(self, visual_features, encoded_prompt, key_padding_mask= None, residual_connection=True):
        # visual_features shape == batch_size, n_channels, features_dim
        # encoded prompt shape == batch_size, n_sequences, features_dim
        temp = torch.cat([visual_features, encoded_prompt], dim=1)
        x = self.self_attn(temp, temp, temp,
                                    key_padding_mask=key_padding_mask,
                                    is_causal=False,
                                    need_weights=False)[0]
        if residual_connection:
            return temp + x
        else:
            return x








if __name__ == '__main__':
    images = torch.randn((4, 32, 256))
    
    embeddings = torch.randn(4,20,256)
    hidden= torch.randn(4,256)
    
    attn = BahdanauAttention(images.shape[-1], hidden.shape[-1], 128)
    
    co_attn = CoAttention(images.shape[-1], embeddings.shape[-1], hidden.shape[-1], 128)
    
    output1 = attn(hidden, images,embeddings)
    
    print(output1[0].shape, output1[1].shape)
    
    output2 = co_attn(hidden, images, embeddings, embeddings)
    
    print(output2[0].shape, output2[1].shape, output2[2].shape)