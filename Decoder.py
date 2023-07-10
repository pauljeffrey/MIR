import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional
from torch import Tensor

"""
    This code implements a decoder transformer layer with an embedding layer, rotary positional embedding layer, self-attention, cross-attention,
    feed forward layers and a causal LM head. The d_model parameter is the size of the input and output embeddings, nhead is the number 
    of attention heads, dim_feedforward is the size of the feed-forward layer hidden state and dropout is the dropout probability.
    
    The tgt parameter represents the input sequence to the decoder. The memory parameter represents the output of the encoder which is 
    used as input during cross-attention. The tgt_mask and memory_mask parameters are optional masks to prevent attention to certain 
    positions in the input sequence.

    The rotary positional embedding layer is implemented using a simple linear layer with ReLU activation. The layer is applied to half 
    of the input dimensions at a time in a rotating fashion. The resulting embeddings are added to the input sequence embeddings.

    The self-attention and cross-attention layers are implemented using PyTorch's built-in MultiheadAttention module. The feed-forward 
    layer is implemented as a simple two-layer MLP with ReLU activation and dropout. The causal LM head is implemented as a linear layer
    applied to the output of the feed-forward layer. It predicts the next token in the sequence based on the current sequence context.

"""


class DecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self,d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        super(DecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,
                 layer_norm_eps, batch_first, norm_first, device=device, dtype=dtype)
        
        
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
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
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)





if __name__ == '__main__':
    layer = DecoderLayer(256, 4)
    inputs = torch.randint(size=(3,4,6))
    outputs = layer(inputs)
    print(outputs)



# class DecoderTransformerLayer(nn.Module):
#     def __init__(self, vocab_size, emb_dim, d_model, nhead, dim_feedforward, dropout=0.1, use_cross_attention=False):
#         super(DecoderTransformerLayer, self).__init__()

#         # Embedding Layer
#         self.embedding = nn.Embedding(vocab_size, emb_dim)

#         # Rotary Positional Embedding Layer
#         self.rotary_pos_enc = nn.Sequential(
#             nn.Linear(d_model//2, d_model//2),
#             nn.ReLU(),
#             nn.Linear(d_model//2, d_model//2)
#         )

#         # Self-Attention Layer
#         self.self_attn_layer_norm = nn.LayerNorm(d_model)
#         self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

#         # Cross-Attention Layer
#         if use_cross_attention:
#             self.cross_attn_layer_norm = nn.LayerNorm(d_model)
#             self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

#         # Feed Forward Layer
#         self.feed_forward_layer_norm = nn.LayerNorm(d_model)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, d_model),
#             nn.Dropout(dropout),
#         )

#         # Causal LM Head
#         self.causal_lm_head = nn.Linear(d_model, vocab_size) #d_model

#         # Dropout Layer
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None):
#         seq_len = tgt.size(1)

#         # Embedding Layer
#         tgt = self.embedding(tgt)

#         # Rotary Positional Encoding Layer
#         pos_emb = torch.zeros(seq_len, tgt.size(-1), device=tgt.device)
#         for i in range(0, seq_len, 2):
#             pos_emb[i:i+2] = self.rotary_pos_enc(torch.arange(tgt.size(-1)//2, device=tgt.device)) \
#                                  .unsqueeze(0) \
#                                  .repeat(2, 1)

#         tgt = tgt + pos_emb.unsqueeze(0)

#         # Self-Attention Layer
#         tgt2 = self.self_attention(
#             tgt, tgt, tgt,
#             attn_mask=tgt_mask,
#             need_weights=False
#         )[0]
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.self_attn_layer_norm(tgt)

#         # Cross-Attention Layer
#         if memory is not None:
#             tgt2 = self.cross_attention(
#                 tgt, memory, memory,
#                 attn_mask=memory_mask,
#                 need_weights=False
#             )[0]
#             tgt = tgt + self.dropout(tgt2)
#             tgt = self.cross_attn_layer_norm(tgt)

#         # Feed Forward Layer
#         tgt2 = self.feed_forward(tgt)
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.feed_forward_layer_norm(tgt)

#         # Causal LM Head
#         tgt = self.causal_lm_head(tgt)

#         return tgt


# class Decoder(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, num_layers, vocab_size, access_features = None):
#         super(Decoder, self).__init__()
#         self.access_visual_features  = access_features
#         self.layers = nn.ModuleList([DecoderTransformerLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

#     def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None):
#         # tgt: [batch_size, tgt_len]
#         # memory: [batch_size, src_len, d_model]

#         for layer_num, layer in enumerate(self.layers):
#             if self.access_visual_features:
#                 memory = memory if self.access_visual_features[layer_num] else None
#                 tgt = layer(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            
#             else:
#                 tgt = layer(tgt, memory= None, tgt_mask=tgt_mask, memory_mask=memory_mask)
                

#         return tgt
