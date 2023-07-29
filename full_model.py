import torch.nn as nn
from Encoder import *
from Decoder import *
from Attention import *
from lstm import *

class MedicalReportGenerator(nn.Module):
    def __init__(self, model_name: str, hidden_layer_size: int, classes: int, fc_in_features: int, semantic_features_dim: int , att_units: int,
                 hidden_dim: int, features_dim: int, k: int, lstm_layers: int, dec_num_layers: int, vocab_size: int, use_topic_per_layer: List[bool], 
                 use_cross_att_per_layer: List[bool], d_model: int, nhead: int, hist_vocab_size: int =None, history_encoder_num_layers: int =None,
                 history_encoder_n_heads: int = None, history_encoder_dim_feedforward: int= None, history_encoder_dmodel: int = None, dim_feedforward: int = 2048, 
                 topic_units: Optional[int]= 0, dropout: float = 0.1, pretrained: bool = True, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu, 
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None,  
                 norm=None, co_attention: bool = True, add_encoder= True, enforce_info= False ):
        
        super(MedicalReportGenerator, self).__init__()
        
        # Image Encoder
        if add_encoder:
            self.encoder = Encoder(model_name, pretrained, hidden_layer_size, classes, fc_in_features,)  
        else:
            self.encoder = None      
        
        if history_encoder_dmodel is not None:
            self.history_encoder = HistoryEncoder(hist_vocab_size, history_encoder_num_layers, history_encoder_dmodel, history_encoder_n_heads, 
                                                  history_encoder_dim_feedforward, dropout, activation, device  )
        else:
            self.history_encoder = None
       
        # Tag embedding module
        if co_attention:
            self.semantic_features_extractor = SemanticFeatureExtractor(classes, semantic_features_dim, k)
            self.attention = CoAttention(features_dim, semantic_features_dim, hidden_dim, att_units)
        
        # Attention Module: Bahdanau Attention
        else:
            self.attention = BahdanauAttention(features_dim, hidden_dim, att_units)
            
        self.co_attention = co_attention

        # LSTM Layer
        self.lstm = SentLSTM( hidden_dim, lstm_layers, features_dim, hidden_layer_size, batch_first, enforce_info)


        # Decoder: Transformer Decoder
        #mem_dim argument changed to features_dim
        # topic_emb changed to hidden_dim of lstm
        self.decoder = MIRDecoder(dec_num_layers, vocab_size, use_topic_per_layer, use_cross_att_per_layer, 
                                  d_model, nhead, features_dim, dim_feedforward, hidden_dim, topic_units, dropout, activation,
                                  layer_norm_eps, batch_first, norm_first, device, dtype, norm)
        
        self.vocab_size = vocab_size
        
        return

    