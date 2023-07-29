import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional, List
from torch import Tensor
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from utils import *

class DecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, topic_emb: Optional[int]= 0,
                 topic_units: Optional[int]= 0,dropout: float = 0.1, mem_dim: Optional[int]=0, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False, use_cross_attention=True,
                 use_topic= True, device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.use_topic = use_topic
        
        key_value_emb_size = d_model + topic_units
        
        self.self_attn = MultiheadAttention(d_model, nhead, kdim= key_value_emb_size, vdim=key_value_emb_size, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        
        if use_cross_attention:
            if mem_dim == 0:
                mem_dim = d_model
            self.multihead_attn = MultiheadAttention(d_model, nhead, kdim=mem_dim, vdim=mem_dim, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        #Implementation of the topic key and value Linear layers
        if use_topic:
            self.topic_key_fc = Linear(topic_emb, topic_units, **factory_kwargs)
            self.topic_value_fc = Linear(topic_emb, topic_units, **factory_kwargs)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
        
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor],
        topic: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        #print("Forward topic shape: ", topic.shape)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, topic, tgt_is_causal)
            if self.use_cross_attention:
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, topic, tgt_is_causal))
            if self.use_cross_attention:
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x
    
     # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor], 
                  topic: Optional[Tensor] = None,
                  is_causal: bool = False) -> Tensor:
        
        if self.use_topic:
            # topic.shape  = (batch_size, topic_emb)
    
            topic_key = self.topic_key_fc(topic)
            topic_value = self.topic_value_fc(topic)
            
            # print(topic_key.shape)
            # print(topic_value.unsqueeze(1).repeat(1,x.shape[1],1).shape)
            # print("Concatenated: ", torch.cat([x, topic_key.unsqueeze(1).repeat(1,x.shape[1],1)], dim=-1).shape)
            
            
            x = self.self_attn(x, 
                               torch.cat([x, topic_key.unsqueeze(1).repeat(1,x.shape[1],1)], dim=-1), 
                               torch.cat([x, topic_value.unsqueeze(1).repeat(1,x.shape[1],1)], dim=-1),
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=False)[0]
        else:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Optional[Tensor],
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)


class MIRDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, num_layers, vocab_size: int, use_topic_per_layer: List[bool], use_cross_att_per_layer: List[bool], d_model: int, 
                 nhead: int, mem_dim: Optional[int], dim_feedforward: int = 2048, topic_emb: Optional[int]= 0, topic_units: Optional[int]= 0, 
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu, layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, norm_first: bool = False, device=None, 
                 dtype=None,  norm=None):
        
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        
        assert num_layers == len(use_topic_per_layer)
        assert num_layers == len(use_cross_att_per_layer)
        self.embed_layer = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, topic_emb = topic_emb if use_topic_per_layer[i] else 0,
                         topic_units= topic_units if use_topic_per_layer[i] else 0, dropout =dropout,mem_dim=mem_dim,
                         activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                         use_cross_attention= use_cross_att_per_layer[i], use_topic= use_topic_per_layer[i], device=device, dtype=dtype) for i in range(num_layers)
            ])
        

        self.num_layers = num_layers
        
        self.norm = norm
        self.causal_lm_head = Linear(d_model, vocab_size,activation="softmax")


    def forward(self, tgt: Tensor, topic: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = True,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        output = self.embed_layer(output)
        
        for mod in self.layers:
            output = mod(output, memory,topic, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        output = self.causal_lm_head(output)
        
        return output
    
    
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


if __name__ == '__main__':
    # layer = DecoderLayer(256, 4)
    # inputs = torch.randint(size=(3,4,6))
    # outputs = layer(inputs)
    # print(outputs)
    
    config = {
        'num_layers': 6,
        'vocab_size': 1000,
        'use_topic_per_layer' : [False,False, False, True, True, True], #[False, True], #
        'use_cross_att_per_layer' : [False, False, False, False, True,True], #[False, True], #
        'd_model': 128, 
        'nhead' : 8, 
        'dim_feedforward' : 256, 
        'topic_emb': 64, 
        'topic_units' : 64, 
        'mem_dim': 64,
        'activation' : F.relu, 
        'layer_norm_eps' : 1e-5, 
        'batch_first': True, 
        'norm_first' : False, 
        'device': 'cpu', 
        'dtype' : None, 
        'norm' : None
    }

    decoder = MIRDecoder(**config)
    # decoder_layer = DecoderLayer(config["d_model"], config["nhead"], config["dim_feedforward"],
    #                        config["topic_emb"], config['topic_units'], mem_dim=config['mem_dim'])
    
    inputs= torch.tensor([[1,5,6,0,0],
                          [13,45, 89,0,10],
                          [0,22, 31, 7, 1],
                          [13,1,0,4,6]])
    
    padding_mask = create_padding_mask(inputs)
    #causal_mask1 = create_causal_masks(inputs)
    causal_mask2 = src_mask(inputs.shape[1])
    
    #print("masks: ", causal_mask2, padding_mask)
    images = torch.randn((4,64, 64 ))
    topic= torch.randn((4,64))
    
    # emb_layer = nn.Embedding(100, 128)
    # inputs = emb_layer(inputs)
    # print("embeddings shape: ", inputs.shape)
    
    # tgt: Tensor, topic: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = True,
    #             memory_is_causal: bool = False
                
    #outputs = decoder_layer(inputs, images, topic,tgt_key_padding_mask=padding_mask, tgt_mask=causal_mask2, tgt_is_causal=True)
    outputs = decoder(inputs,topic,images,tgt_key_padding_mask=padding_mask, tgt_mask=causal_mask2, tgt_is_causal=True)
    print("Output shape: ", outputs.shape)
    #print(decoder)


