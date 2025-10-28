
import torch
from torch import nn
from transformer.transfomer_encoder import TransformerEncoder


class ImageEncoder(nn.Module):
    
    def __init__(self, input_resolution, kernel_size, hidden, transformer_count, attn_head_count, out_dim):
        super().__init__()        
        
        self.convolution = nn.Conv2d(in_channels=3, out_channels=hidden, kernel_size=kernel_size, stride=kernel_size)

        scale = hidden ** -0.5  #=sqrt(1/hidden)

        self.class_embedding = nn.Parameter(scale * torch.randn(hidden))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // kernel_size) ** 2 + 1, hidden))
        self.pre_norm = nn.LayerNorm(hidden)

        self.transformer = nn.Sequential(*[TransformerEncoder(hidden, attn_head_count)] * transformer_count)
        self.post_norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, out_dim)


    def forward(self, image):

        x = self.convolution(image)  # shape = [*, width, grid, grid]
        x = x.flatten(start_dim=-2, end_dim=-1) # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        class_embedding = torch.stack([self.class_embedding] * x.shape[0], dim=0).unsqueeze(1)
        x = torch.cat([class_embedding, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding

        x = self.pre_norm.forward(x)

        x = self.transformer.forward(x)
        
        x = self.post_norm.forward(x[:, 0, :])

        x = self.projection.forward(x)

        return x
