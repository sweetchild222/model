import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .residual_norm import ResidualNorm
from .positionwise_feed_forward import PositionwiseFeedForward


class TransformerEncorder(nn.Module):
    
    def __init__(self, hidden, attention_head_count, feed_forward_hidden, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(head_count=attention_head_count, d_model=hidden)
        self.attention_residual_norm = ResidualNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.feed_forward_residual_norm = ResidualNorm(hidden)
        

    def forward(self, x, mask=None):

        x = self.attention.forward(query=x, key=x, value=x, mask=mask)

        x = self.attention_residual_norm.forward(x)

        x = self.dropout.forward(x)

        x = self.feed_forward.forward(x)
        
        x = self.feed_forward_residual_norm.forward(x)
    
        return x
