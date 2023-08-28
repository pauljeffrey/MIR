import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional, List,Tuple
from torch import Tensor
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from utils import *
#from xformers.components.positional_embedding import XFORMERS.COMPONENTS.POSITIONAL_EMBEDDING.ROTARY
#from rotary_embedding import RotaryEmbedding

import torch
import torch.nn as nn
import math
from rotary_embedding_torch import RotaryEmbedding

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

    def __init__(self, pos_emb: RotaryEmbedding, d_model: int, nhead: int, dim_feedforward: int = 2048, topic_emb: Optional[int]= 0,
                 topic_units: Optional[int]= 0,dropout: float = 0.1, features_dim: Optional[int]=0, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False, use_cross_attention=True,
                 use_topic= True, use_prompt=False, device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.use_topic = use_topic
        
        key_value_emb_size = d_model + topic_units
        
        self.pos_emb  = pos_emb
        self.n_head = nhead
        self.d_model = d_model
        self.use_prompt = use_prompt
        
        self.self_attn = MultiheadAttention(d_model, nhead, kdim= key_value_emb_size, vdim=key_value_emb_size, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        
        if use_cross_attention:
            if use_prompt:
                self.multihead_attn = MultiheadAttention(d_model, nhead, kdim=d_model, vdim=d_model, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
            else:
                self.multihead_attn = MultiheadAttention(d_model, nhead, kdim=features_dim, vdim=features_dim, dropout=dropout, batch_first=batch_first,
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
        
        #print( tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        if torch.any(torch.isnan(tgt)):
            print(f"The targets passed as input to the decoder has nan values..")
        
        x = tgt
        #print("Forward topic shape: ", topic.shape)
        #print("Got here")
        if torch.any(torch.isnan(x)):
            print(f"The input x gotten from targets passed to the decoder has nan values..")
            
        # print("min max of indication: ", torch.min(memory[0]), torch.max(memory[0]))
        # print("min max of indication: ", torch.min(memory[1]), torch.max(memory[1]))
        # print("min max of tgt: ", torch.min(x), torch.max(x))
        # print("min max of topic: ", torch.min(topic), torch.max(topic))
        
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, topic, tgt_is_causal)
            
            if self.use_cross_attention:
                prompt, weighted_images = memory[0], memory[1]
                if self.use_prompt:
                    x = x + self._mha_block(self.norm2(x), prompt, memory_mask, memory_key_padding_mask, memory_is_causal)
                else:    
                    x = x + self._mha_block(self.norm2(x), weighted_images, memory_mask, key_padding_mask=None, is_causal=memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, topic, tgt_is_causal))
            
            if self.use_cross_attention:
                prompt, weighted_images = memory[0], memory[1]
                if self.use_prompt:
                    #print("Image shape; ", memory_mask, memory_key_padding_mask)
                    x = self.norm2(x + self._mha_block(x, prompt, memory_mask, key_padding_mask=memory_key_padding_mask, is_causal=memory_is_causal))
                else:
                    memory_key_padding_mask = None
                    #print("Image shape; ", memory_mask.shape)
                    x = self.norm2(x + self._mha_block(x, weighted_images, memory_mask, key_padding_mask=memory_key_padding_mask, is_causal=memory_is_causal))
            
                
            x = self.norm3(x + self._ff_block(x))
            

        return x
    
     # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor], 
                  topic: Optional[Tensor] = None,
                  is_causal: bool = False) -> Tensor:
        
        
        b, seq_l = x.shape[:2]
        
        q= x.reshape(-1, seq_l, self.n_head, int(self.d_model/self.n_head)).permute(0,2,1,3)
        k = x.reshape(-1, seq_l, self.n_head, int(self.d_model/self.n_head)).permute(0,2,1,3)
        
        #apply positional rotary embedding:
        q = self.pos_emb.rotate_queries_or_keys(q)
        k = self.pos_emb.rotate_queries_or_keys(k)
        
        # Reshape q, k to normal
        q = q.permute(0,2,1,3).reshape(b, seq_l, self.d_model)
        k = k.permute(0,2,1,3).reshape(b, seq_l, self.d_model)
        
        if torch.any(torch.isnan(q)):
            print("query contains nan values..")
            
        if torch.any(torch.isnan(k)):
            print("key contains nan values..")
            
        if torch.any(torch.isnan(x)):
            print("x contains nan values..")
            
        # Apply topic vector to each token
        if self.use_topic:
            # topic.shape  = (batch_size, topic_emb)
    
            topic_key = self.topic_key_fc(topic)
            topic_value = self.topic_value_fc(topic)
            
            if torch.any(torch.isnan(topic_key)) or torch.any(torch.isnan(topic_value)):
                print("Topic values inside decoder layers is affected...")
            
            # print(topic_key.shape)
            # print(topic_value.unsqueeze(1).repeat(1,x.shape[1],1).shape)
            # print("Concatenated: ", torch.cat([x, topic_key.unsqueeze(1).repeat(1,x.shape[1],1)], dim=-1).shape)
            #print("Use Topic: ",x.shape, topic_key.shape, topic_value.shape)
            #print(k.shape, topic_key.repeat(1,x.shape[1],1).shape)
            #print("Use topic", attn_mask.shape, is_causal)
            x = self.self_attn(q, 
                               torch.cat([k, topic_key.repeat(1,x.shape[1],1)], dim=-1), 
                               torch.cat([x, topic_value.repeat(1,x.shape[1],1)], dim=-1),
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=False)[0]
        else:
            #print("Don't use topic", attn_mask, is_causal)
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=False)[0]
            
        if torch.any(torch.isnan(x)):
                print("Self attention is affected.. ")
                
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if torch.any(torch.isnan(x)):
                print("Feed Forward block is affected..")
        return self.dropout2(x)
    
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Optional[Tensor],
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        #print("mha shape: ",x.shape, mem.shape)
        b, seq_l = x.shape[:2]
        if self.use_prompt:
            #print("Prompt shape: ", mem.shape)
            prompt_seq_l, prompt_d_model = mem.shape[1], mem.shape[-1]
            #print(prompt_seq_l, prompt_d_model, self.n_head)
            assert prompt_d_model % self.n_head == 0
            
            q= x.reshape(-1, seq_l, self.n_head, int(self.d_model/self.n_head)).permute(0,2,1,3)
            k = mem.reshape(-1, prompt_seq_l, self.n_head, int(prompt_d_model/self.n_head)).permute(0,2,1,3)
            
            #apply positional rotary embedding:
            #print(k.shape, q.shape)
            q = self.pos_emb.rotate_queries_or_keys(q)
            k = self.pos_emb.rotate_queries_or_keys(k)
            
            # Reshape q, k to normal
            q = q.permute(0,2,1,3).reshape(b, seq_l, self.d_model)
            k = k.permute(0,2,1,3).reshape(b, prompt_seq_l, prompt_d_model)
            
            if torch.any(torch.isnan(x)):
                print("cross attention x contains nan values.")
        
            if torch.any(torch.isnan(q)):
                print("cross attention query contains nan values..")
                
            if torch.any(torch.isnan(k)):
                print("cross attention key contains nan values..")
                
            if torch.any(torch.isnan(mem)):
                print("cross attention mem contains nan values..")
                
            x = self.multihead_attn(q, k, mem,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    is_causal=is_causal,
                                    need_weights=False)[0]
        else:
            x= x.reshape(-1, seq_l, self.n_head, int(self.d_model/self.n_head)).permute(0,2,1,3)
            #apply positional rotary embedding:
            x= self.pos_emb.rotate_queries_or_keys(x)
            
            if torch.any(torch.isnan(x)):
                print("cross attention x contains nan values.")
                
            if torch.any(torch.isnan(mem)):
                print("cross attention mem contains nan values..")
        
            # Reshape q to normal
            x = x.permute(0,2,1,3).reshape(b, seq_l, self.d_model)
            
            x = self.multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    is_causal=is_causal,
                                    need_weights=False)[0]
            
        if torch.any(torch.isnan(x)):
            print("Cross attention is affected")
            
        return self.dropout2(x)
    
    


class MIRDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, num_layers, vocab_size: int, use_topic_per_layer: List[bool], use_prompt_per_layer: List[bool], use_cross_att_per_layer: List[bool], d_model: int, 
                 nhead: int, mem_dim: Optional[int], dim_feedforward: int = 2048, topic_emb: Optional[int]= 0, topic_units: Optional[int]= 0, 
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu, layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, norm_first: bool = False, device=None, 
                 dtype=None,  norm=True):
        
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        
        assert num_layers == len(use_topic_per_layer)
        assert num_layers == len(use_cross_att_per_layer)
        self.embed_layer = nn.Embedding(vocab_size, d_model)
        self.rotary_embed =  RotaryEmbedding(
            dim = 32,
            #use_xpos = True   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
        )
        
        #self.use_prompt_per_layer = use_prompt_per_layer
        self.layers = nn.ModuleList([
            DecoderLayer(self.rotary_embed, d_model, nhead, dim_feedforward, topic_emb = topic_emb if use_topic_per_layer[i] else 0,
                         topic_units= topic_units if use_topic_per_layer[i] else 0, dropout =dropout,features_dim=mem_dim,
                         activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                         use_cross_attention= use_cross_att_per_layer[i], use_topic= use_topic_per_layer[i], use_prompt=use_prompt_per_layer[i],device=device, dtype=dtype) for i in range(num_layers)
            ])
        

        self.num_layers = num_layers
        
        self.norm = norm
        self.causal_lm_head = Linear(d_model, vocab_size)
        


    def forward(self, tgt: Tensor, topic: Tensor=None, memory: Tensor=None, tgt_mask: Optional[Tensor] = None,
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
        
        # tgt_key_padding_mask = self.get_padding_mask(tgt)
        # #causal_mask1 = create_causal_masks(inputs)
        # tgt_mask = self.get_causal_mask(tgt)
        
        output = tgt
        output = self.embed_layer(output)
        
        print("Inside decoder, the embedding output min and max values: ", torch.min(output) , torch.max(output))
        
        if torch.any(torch.isnan(output)):
            print("Inside decoder, after embedding layer: ", output)
            
        for ind, mod in enumerate(self.layers):
            output = mod(output, memory, topic, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal) #memory_is_causal=memory_is_causal,use_prompt=self.use_prompt_per_layer[ind]

            if torch.any(torch.isnan(output)):
                print("Affected Decoder Layer is : ", ind)
        
        if self.norm is not None:
            output = self.norm(output)

        output = self.causal_lm_head(output)
        
        return output
    
    def get_causal_mask(self, input):
        return src_mask(input.shape[1])
    
    def get_padding_mask(self, input):
        return create_padding_mask(input)
    
    
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
        'use_prompt_per_layer': [False, False, False,True, True, False],
        'd_model': 256, 
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
    #print(padding_mask)
    #causal_mask1 = create_causal_masks(inputs)
    causal_mask2 = src_mask(inputs.shape[1])
    
    #print("masks: ", causal_mask2, padding_mask)
    images = torch.randn((4,64, 64 ))
    topic= torch.randn((4,1,64))
    
    #embeddings = torch.randn(4,20,256)
    hist = torch.randint(0,5, (4,20))
    mem_mask= create_padding_mask(hist)
    emb_layer = nn.Embedding(1000, 256)
    embeddings = emb_layer(hist)
    # inputs = emb_layer(inputs)
    # print("embeddings shape: ", inputs.shape)
    
    # tgt: Tensor, topic: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = True,
    #             memory_is_causal: bool = False
                
    #outputs = decoder_layer(inputs, images, topic,tgt_key_padding_mask=padding_mask, tgt_mask=causal_mask2, tgt_is_causal=True)
    outputs = decoder(inputs,topic,(embeddings,images),tgt_key_padding_mask=padding_mask,
                      memory_key_padding_mask=mem_mask, tgt_mask=causal_mask2, tgt_is_causal=True)
    print("Output shape: ", outputs.shape)
    #print(decoder)
    
    
    images = torch.randn((4,64, 128 ))
    encoded = torch.randn((4,20, 128))
    #mha = MHA(128,2,)
    
    
   
    #print("Rotary embedding: ", torch.transpose(q[1,1,1,:4],0,1), torch.transpose(q[2,1,1,:4], 0,1))
    # emb_layer = RotaryEmbedding(256)
    
    # print(emb_layer(embeddings, torch.range(0,20)).shape)

# import torch


# # instantiate the positional embedding in your transformer and pass to all your attention layers

# rotary_emb = RotaryEmbedding(dim = 32)

# # mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

# q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
# k = torch.randn(1, 8, 1024, 64) # keys

# # apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

# q = rotary_emb.rotate_queries_or_keys(q)
# k = rotary_emb.rotate_queries_or_keys(k)
# print("q shape: ", torch.permute(q, (0,2,1,3)).shape)
# q =torch.permute(q, (0,2,1,3)).reshape(1,1024, 64*8)
# k = torch.permute(k,(0,2,1,3)).reshape(1,1024, 8*64)
# # then do your attention with your queries (q) and keys (k) as usual

# v = torch.randn(1, 1024, 64*8)
# att = nn.MultiheadAttention(64*8, 8)
# out = att(q,k,v)
# print("q shape", q.shape)
# print("k shape", k.shape)
# # then do your attention with your queries (q) and keys (k) as usual