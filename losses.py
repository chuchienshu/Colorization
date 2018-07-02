import torch
import torch.nn as nn
import math

class CE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, predict, target):
        n, c, h, w = target.data.shape

        predict = predict.permute(0,2,3,1).contiguous().view(n*h*w, -1)
        target = target.permute(0,2,3,1).contiguous().view(n*h*w, -1)
        #[262144, 313]
        return self.loss(predict, torch.max(target, 1)[1])
