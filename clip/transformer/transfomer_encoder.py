from torch import nn
from .multi_head_attention import MultiHeadAttention


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, head_count, mask=None):
        super().__init__()
        
        self.mask = mask
        self.attention = MultiHeadAttention(head_count, d_model)
        self.attention_norm = nn.LayerNorm(d_model)

        self.position_wise = nn.Sequential(nn.Linear(d_model, d_model * 4),
                                           nn.GELU(),
                                           nn.Linear(d_model * 4, d_model))

        self.position_wise_norm = nn.LayerNorm(d_model)
        

    def forward(self, x):
        
        mask = self.mask.to(x.device) if self.mask is not None else None
        
        x = x + self.attention.forward(x, x, x, mask=mask) #attention and residual Connection
        
        x = self.attention_norm.forward(x)

        x = x + self.position_wise.forward(x) #position_wise and residual Connection

        x = self.position_wise_norm.forward(x)

        return x
    
