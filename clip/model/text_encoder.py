import torch
from torch import nn
from .transformer import Transformer


class TextEncoder(nn.Module):
    def __init__(self, embed_dim, context_length, vocab_size, trans_d_model, trans_heads, trans_layers):
        super().__init__()

        attn_mask = self.build_attention_mask(context_length)
        
        self.transformer = Transformer(trans_d_model, trans_heads, trans_layers, attn_mask)
        
        self.token_embedding = nn.Embedding(vocab_size, trans_d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.2)

        self.positional_embedding = nn.Parameter(torch.empty(context_length, trans_d_model))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.norm_final = nn.LayerNorm(trans_d_model)

        self.projection = nn.Parameter(torch.empty(trans_d_model, embed_dim))
        nn.init.normal_(self.projection, std=trans_d_model ** -0.5)


    def build_attention_mask(self, context_length):
        
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)

        return mask


    def forward(self, text):

        #[batch_size, n_ctx, d_model]
        x = self.token_embedding(text)

        x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        x = torch.matmul(x, self.projection)

        return x
