import torch
from torch import nn

from transformer.transfomer_encoder import TransformerEncoder
from positional_encoding import PositionalEncoding


class TextEncoder(nn.Module):

    def __init__(self, seq_len, vocab_size, hidden, transformer_count, attn_head_count, out_dim):
        super().__init__()

        mask = torch.ones((seq_len, seq_len), dtype=torch.bool).tril()
        
        self.transformer = nn.Sequential(*[TransformerEncoder(hidden, attn_head_count, mask)] * transformer_count)
        
        self.token_embedding = nn.Embedding(vocab_size, hidden)
                
        self.positional_encoding = PositionalEncoding(hidden, seq_len)
        
        self.norm = nn.LayerNorm(hidden)

        self.projection = nn.Linear(hidden, out_dim)


    def forward(self, text):

        x = self.token_embedding.forward(text) + self.positional_encoding.forward(text)

        x = self.transformer.forward(x)

        x = self.norm(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
         
        x = self.projection.forward(x)

        return x


