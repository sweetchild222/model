import torch.nn as nn
from .scale_dot_attention import ScaleDotAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, head_count, d_model):
        super().__init__()

        assert d_model % head_count == 0

        # We assume d_v always equals d_k
        self.depth = d_model // head_count
        self.head_count = head_count

        self.query_liner = nn.Linear(d_model, d_model)
        self.key_liner = nn.Linear(d_model, d_model)
        self.value_liner = nn.Linear(d_model, d_model)

        self.scale_dot_attention = ScaleDotAttention()
        self.output_projection = nn.Linear(d_model, d_model)
        

    def split_heads(self, x):

        batch_size = x.size(0)

        x = x.view(batch_size, -1, self.head_count, self.depth)

        return x.transpose(1, 2)


    def forward(self, query, key, value, mask=None):
        
        batch_size = query.size(0)

        query = self.query_liner.forward(query)
        key = self.key_liner.forward(key)
        value = self.value_liner.forward(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        x = self.scale_dot_attention(query, key, value, mask=mask)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count * self.depth)

        return self.output_projection.forward(x)
