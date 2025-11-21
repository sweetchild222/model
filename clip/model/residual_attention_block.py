import torch
from torch import nn
from collections import OrderedDict


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, n_head)
        self.norm_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))]))
        
        self.norm_2 = nn.LayerNorm(d_model)

        self.attn_mask = attn_mask


    def attention_forward(self, x: torch.Tensor):
    
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attention(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):

        x = x + self.attention_forward(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x
