import torch.nn as nn
from CNN_Encoder import *
from Decoder import *
from Attention import *
from lstm import *

class MedicalReportGenerator(nn.Module):
    def __init__(self, model_name: str, hidden_layer_size: int, classes: int, semantic_features_dim: int , att_units: int, 
                 hidden_dim: int, features_dim: int, k: int, lstm_layers: int, dec_num_layers: int, vocab_size: int, use_topic_per_layer: List[bool], 
                 use_cross_att_per_layer: List[bool], use_prompt_per_layer: List[bool], d_model: int, nhead: int, history_encoder_num_layers: int =None, model_dict=None,
                 history_encoder_n_heads: int = None, history_encoder_dim_feedforward: int= None, dim_feedforward: int = 4096, 
                 topic_units: Optional[int]= 0, dropout: float = 0.1, pretrained: bool = True, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu, 
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False, device="cuda", dtype=None,  
                 norm=None, co_attention: bool = True, add_encoder= True, enforce_info= False, pa_nhead= 1,use_residual = True,use_history=True ):
        # history_encoder_dmodel: int = None,
        super(MedicalReportGenerator, self).__init__()
        # Image Encoder
        if add_encoder:
            self.encoder = VisualFeatureExtractor(model_name).to(device) #, pretrained, hidden_layer_size, classes, model_dict)  
        else:
            self.encoder = None      
         
        if use_history is not None:
            self.history_encoder = HistoryEncoder(history_encoder_num_layers, d_model, history_encoder_n_heads, 
                                                  history_encoder_dim_feedforward, dropout, activation, device  ).to(device)
        else:
            self.history_encoder = None
       
        # Tag embedding module
        if co_attention:
            self.semantic_features_extractor = SemanticFeatureExtractor(classes, semantic_features_dim, k).to(device)
            self.attention = CoAttention(features_dim, semantic_features_dim, hidden_dim, att_units, d_model)
        
        # Attention Module: Bahdanau Attention
        else:
            self.attention = BahdanauAttention(features_dim, hidden_dim, att_units).to(device)
            self.semantic_features_extractor = None
            
        self.co_attention = co_attention

        # LSTM Layer
        #hidden_size, num_layers, features_dim, hidden_layer_size, d_model
        self.sent_lstm = SentLSTM( hidden_dim, lstm_layers, features_dim, hidden_layer_size, d_model, batch_first, enforce_info).to(device)

        # promtp_image attn
        #self.prompt_attention = MHA(features_dim, pa_nhead, dropout=dropout, use_residual=use_residual, batch_first=batch_first, device=device, dtype=dtype)
        
        # Decoder: Transformer Decoder
        #mem_dim argument changed to features_dim
        # topic_emb changed to hidden_dim of lstm
        
        self.decoder = MIRDecoder(dec_num_layers, vocab_size, use_topic_per_layer, use_cross_att_per_layer, use_prompt_per_layer,
                                  d_model, nhead, features_dim, dim_feedforward, hidden_dim, topic_units, dropout, activation,
                                  layer_norm_eps, batch_first, norm_first, device, dtype, norm).to(device)
        
        self.vocab_size = vocab_size
        
        return

    
    if __name__ == '__main__':
        # model = MedicalReportGenerator(model_name: str, hidden_layer_size: int, classes: int, semantic_features_dim: int , att_units: int, pa_nhead: int,
        #          hidden_dim: int, features_dim: int, k: int, lstm_layers: int, dec_num_layers: int, vocab_size: int, use_topic_per_layer: List[bool], 
        #          use_cross_att_per_layer: List[bool], d_model: int, nhead: int, history_encoder_num_layers: int =None, model_dict=None,
        #          history_encoder_n_heads: int = None, history_encoder_dim_feedforward: int= None, history_encoder_dmodel: int = None, dim_feedforward: int = 2048, 
        #          topic_units: Optional[int]= 0, dropout: float = 0.1, pretrained: bool = True, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu, 
        #          layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None,  
        #          norm=None, co_attention: bool = True, add_encoder= True, enforce_info= False, use_residual = True)
        pass