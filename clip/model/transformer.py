from torch import nn
from .residual_attention_block import ResidualAttentionBlock


class Transformer(nn.Module):
    def __init__(self, d_model, heads, layers, attn_mask):
        super().__init__()
        
        attn_std = d_model ** -0.5
        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * d_model) ** -0.5

        self.sequential = nn.Sequential()

        for _ in range(layers):

            block = ResidualAttentionBlock(d_model, heads, attn_mask)

            nn.init.normal_(block.attention.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attention.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            self.sequential.append(block)


    def forward(self, x):
        
        return self.sequential(x)
