import torch.nn as nn
from .positional_encoding import PositionalEncoding


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.positional_embedding = PositionalEncoding(d_model=self.token_embedding.embedding_dim)
        self.segment_embedding = nn.Embedding(3, self.token_embedding.embedding_dim, padding_idx=0)            
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, segment):

        x = self.token_embedding.forward(x) + self.positional_embedding.forward(x) + self.segment_embedding.forward(segment)

        return self.dropout.forward(x)
