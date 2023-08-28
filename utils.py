import torch
import torch.nn as nn
import numpy as np


def src_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-1e8')).masked_fill(mask == 1, float(0.0))
    return mask


def create_masks(inputs):
    # Create padding mask
    padding_mask = torch.tensor((inputs != 0)).unsqueeze(1).unsqueeze(2)
    
    # Create causal mask
    causal_mask = src_mask(inputs.shape[1])
    #print(torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1))
    # Combine padding and causal masks
    masks = padding_mask & causal_mask
    
    masks = masks.float().masked_fill(masks == 0, float('-1e8')).masked_fill(masks == 1, float(0.0))
    return masks


def create_padding_mask(inputs):
    """ False for non-padding element otherwise, True"""
    # if type(inputs) == np._NdArraySubClass:
    #     print("yes")
    if type(inputs) != torch.Tensor:
        mask = torch.tensor((inputs == 0))
        
    else:
        mask = (inputs == 0)#.unsqueeze(1).unsqueeze(2)
    mask = mask.float().masked_fill(mask == 1, float('-1e8'))#.masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == '__main__':
    inputs = np.array([[5, 12, 8, 0, 0],[32, 15, 16, 0, 7],[0,0, 15,32, 64]])
   
    #print(create_causal_masks(inputs))
    # print(generate_square_subsequent_mask(5))
    # print()
    mask = create_padding_mask(inputs)
    # # print()
    
    # print(inputs)
    
    q = torch.randn( 3,5, 10) # source sequence length 3, batch size 1, embedding size 10
    w= torch.randn(1,3)
    if type(w) == torch.Tensor:
        print("YEEEEEEEEEEEEEEEEEEEEEEEEESSSSSSSSSSSSSSSSSSSS")
    attn = nn.MultiheadAttention(10, 1, batch_first=True) # embedding size 10, one head
    #attn(q, q, q) # self attention
    
    # def src_mask(sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask
    
    out = attn(q, q, q, key_padding_mask= mask, attn_mask=src_mask(5))#, 
    print(out[1][0], out[0][0])
    print(out[1].shape, out[0].shape) # attention output weights
    print(src_mask(5).dtype)
    print(mask.dtype)
    
    