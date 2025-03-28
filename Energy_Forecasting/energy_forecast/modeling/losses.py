import torch
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, preds, target):
        return torch.sqrt(self.mse(preds, target))