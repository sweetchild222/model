import torch.nn as nn
import torch

import math


class ScaleDotAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        if mask is not None:            
            scores += scores.masked_fill(mask.logical_not(), float("-inf"))

        attention_weigth = nn.Softmax(dim=-1).forward(scores)

        return torch.matmul(attention_weigth, value)
