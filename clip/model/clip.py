import numpy as np
import torch
from torch import nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class CLIP(nn.Module):
    def __init__(self, embed_dim, img_resolution, img_width, context_length, vocab_size, trans_d_model, trans_heads, trans_layers):
        super().__init__()
        
        self.image_encoder = ImageEncoder(img_resolution, img_width, embed_dim)

        self.text_encoder = TextEncoder(embed_dim, context_length, vocab_size, trans_d_model, trans_heads, trans_layers)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def forward(self, image, text):
        
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)

        return image_features, text_features 
