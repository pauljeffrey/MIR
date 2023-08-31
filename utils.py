import torch
import torch.nn as nn
import numpy as np
import os

def load(model, save_dir):
    successful_loads = 0
    if model.encoder is not None and os.path.exists(os.path.join(save_dir, "encoder.pt")):
        model.encoder.load_state_dict(torch.load(os.path.join(save_dir, "encoder.pt")))
        successful_loads += 1
        
    if model.history_encoder is not None and os.path.exists(os.path.join(save_dir, "prompt_encoder.pt")):
        model.history_encoder.load_state_dict(torch.load(os.path.join(save_dir, "prompt_encoder.pt")))
        successful_loads += 1
        
    if model.semantic_features_extractor is not None and os.path.exists(os.path.join(save_dir, "semantic_features_extractor.pt")):
        model.semantic_features_extractor.load_state_dict(torch.load(os.path.join(save_dir, "semantic_features_extractor.pt")))
        successful_loads += 1
        
    if os.path.exists(os.path.join(save_dir, "attention.pt")):
        model.attention.load_state_dict(torch.load(os.path.join(save_dir, "attention.pt")))
        successful_loads += 1
        
    if os.path.exists(os.path.join(save_dir, "sent_lstm.pt")):
        model.sent_lstm.load_state_dict(torch.load(os.path.join(save_dir, "sent_lstm.pt")))
        successful_loads += 1
        
    if os.path.exists(os.path.join(save_dir, "decoder.pt")):
        model.decoder.load_state_dict(torch.load(os.path.join(save_dir, "decoder.pt")))
        successful_loads += 1
        
    print(f"Successfully loaded {successful_loads} sub models.")
    
    return model

def src_mask(sz, mode="float"):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    
    if mode == "float":
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) 
    return mask


def create_masks(inputs):
    # Create padding mask
    padding_mask = torch.tensor((inputs != 0)).unsqueeze(1).unsqueeze(2)
    
    # Create causal mask
    causal_mask = src_mask(inputs.shape[1], "bool")
    #print(torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1))
    # Combine padding and causal masks
    masks = padding_mask & causal_mask
    
    masks = masks.float().masked_fill(masks == 0, float('-inf')).masked_fill(masks == 1, float(0.0))
    return masks


def create_padding_mask(inputs):
    """ False for non-padding element otherwise, True"""
    # if type(inputs) == np._NdArraySubClass:
    #     print("yes")
    if type(inputs) != torch.Tensor:
        mask = torch.tensor((inputs == 0))
        
    else:
        mask = (inputs == 0)#.unsqueeze(1).unsqueeze(2)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == '__main__':
    inputs = np.array([[5, 0, 0, 7, 15, 5, 8, 10,0, 0],[32, 15, 16, 5, 7, 0, 0, 0, 0,0],[0,0, 15,32, 64, 4,7,9,14,11]])
   
    #print(create_causal_masks(inputs))
    # print(generate_square_subsequent_mask(5))
    # print()
    mask = create_padding_mask(inputs)
    # # print()
    
    # print(inputs)
    masks = create_masks(inputs)
    
    q = torch.randn( 3,10, 10) # source sequence length 3, batch size 1, embedding size 10
    w= torch.randn(1,3)

    attn = nn.MultiheadAttention(10, 2, batch_first=True) # embedding size 10, one head
    #attn(q, q, q) # self attention
    
    # def src_mask(sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask
    
    out = attn(q, q, q, key_padding_mask= None, attn_mask=src_mask(10))#, 
    #print(torch.min(out[1][0]), torch.max(out[1][0]))
    print(out[1][0], out[0][0])
    # out = attn(q, q, q, key_padding_mask= None, attn_mask=masks)#, 
    # #print(torch.min(out[1][0]), torch.max(out[1][0]))
    # print("Output of create_masks: ")
    # print(out[1][0], out[0][0])
    #print(out[1].shape, out[0].shape) # attention output weights
    # print(src_mask(5, "bool"))
    # print(mask)
    
    