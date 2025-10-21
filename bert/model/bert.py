import torch.nn as nn

from .transformer.transformer_encorder import TransformerEncorder
from .embedding.bert_embedding import BERTEmbedding
from .output.bert_output import BERTOutput

class BERT(nn.Module):

    def __init__(self, vocab_size, embed_size, encoder_count=12, attention_head_count=12, dropout=0.1):        
        super().__init__()
        
        self.embedding = BERTEmbedding(vocab_size, embed_size)
        
        self.transformer_encorders = nn.ModuleList(
            [TransformerEncorder(embed_size, attention_head_count, embed_size * 4, dropout)] * encoder_count)

        self.output = BERTOutput(vocab_size, embed_size)


    def forward(self, x, segment, mask=None):

        x = self.embedding.forward(x, segment)
        
        for transformer_encorder in self.transformer_encorders:
            x = transformer_encorder.forward(x, mask)
        
        return self.output.forward(x)
