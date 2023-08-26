import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        #self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()#nn.BCEWithLogitsLoss()
        
    
    def forward(self,y1_true, y2_true, y1_pred, y2_pred,  eval=False): #with_logits=False,
        # if with_logits:
        #     y2_pred = y2_pred.softmax(dim=2)
        y1_mask = y1_true.ne(0)
        y2_mask = y2_true.ne(0)
        
        #Set y2 label to long and y label to float, leave the pred in float32
        # print(y1_true.dtype, y2_true.dtype, y1_pred.dtype, y2_pred.dtype)
        # print("masks shape: ", y1_mask.shape, y2_mask.shape)
        # print("Outpoutshape: ", y1_pred.shape, y1_true.shape)
        # print("Outpoutshape: ", y2_pred.shape, y2_true.shape)
        #y2_pred = y2_pred[y2_mask].permute(0,2,1) # shape == (batch_size, n_classes, seq_len)
        #print("output shape: ", y2_pred.shape, y2_true[y2_mask])
        # if torch.any(torch.isnan(y1_pred)):
        #     print("Y1 pred is nan")
            
        # if torch.any(torch.isnan(y2_pred)):
        #     print("y2_pred is nan")
        
        sparse_loss = self.cross_entropy(y2_pred[y2_mask], y2_true[y2_mask])
        bce_loss = self.bce(y1_pred[y1_mask], y1_true[y1_mask])
        
        # print("sparse_loss: ", sparse_loss)
        # print("stop loss: ", bce_loss)
        #print(bce_loss, sparse_loss)
        if eval:
            return bce_loss , sparse_loss
        else:
            return bce_loss + sparse_loss
        

class CustomBCELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomBCELoss, self).__init__()
        #self.alpha = alpha
        self.bce = nn.BCELoss()#nn.BCEWithLogitsLoss()
        
    
    def forward(self,label_true, label_pred):
        label_mask = label_true.ne(-1)
        label_true = label_true.to(torch.float32)
        label_pred  = label_pred.to(torch.float32)
        
        # if torch.any(torch.isnan(label_pred)):
        #     print("label pred is nan")
            
        #print(label_pred, label_true)
        label_loss = self.bce(label_pred[label_mask], label_true[label_mask])
        # if torch.any(torch.isnan(label_loss)):
        #     print("label_loss is nan")
            
        return label_loss



if __name__ == "__main__":
    # Example of target with class indices
    #loss = nn.CrossEntropyLoss()
    input = torch.randn(10, 5, 10, dtype=torch.float)#.softmax(dim=2)
    #target = torch.empty(10,5, dtype=torch.float).random_(10)
    target = torch.tensor([[5., 8., 0., 1., 1.],
        [8., 2., 2., 7., 1.],
        [3., 7., 9., 3., 0.],
        [5., 9., 3., 9., 8.],
        [7., 6., 5., 8., 2.],
        [3., 2., 4., 9., 5.],
        [8., 2., 3., 5., 2.],
        [2., 4., 4., 5., 5.],
        [4., 9., 6., 2., 4.],
        [9., 9., 1., 9., 4.]],dtype=torch.long)
    
    
    print(input.shape, target.shape)
    
    # Example of target with class probabilities
    input2 = torch.randn(10, 1)#.softmax(dim=1)
    target2 = torch.randint(-1,1, (10,1), dtype=torch.float)

    # loss = nn.CrossEntropyLoss()
    # print(loss(input, target))
    
    c_loss = CustomLoss()
    output = c_loss( target2, target, input2, input)
    print(output)