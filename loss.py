import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        #self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    
    def forward(self, y1_true, y2_true, y1_pred, y2_pred, with_logits=True):
        if with_logits:
            y2_pred = y2_pred.softmax(dim=2)
        y1_mask = y1_true.ne(-1)
        y2_mask = y2_true.ne(0)
        sparse_loss = self.cross_entropy(y2_pred[y2_mask], y2_true[y2_mask])
        bce_loss = self.bce(y1_pred[y1_mask], y1_true[y1_mask])
        print(bce_loss)
        
        return bce_loss + sparse_loss





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