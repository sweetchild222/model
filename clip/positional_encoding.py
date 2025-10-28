import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len):
        super().__init__()
        
        pe = torch.zeros(seq_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = self.pe[:, :x.size(1)]
        x = x.squeeze()
        
        return x
