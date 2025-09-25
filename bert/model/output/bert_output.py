import torch.nn as nn

from .next_type_output import NextTypeOutput
from .mask_output import MaskOutput


class BERTOutput(nn.Module):
    
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.next_type_ouput = NextTypeOutput(embed_size)
        self.mask_output = MaskOutput(embed_size, vocab_size)


    def forward(self, x):

        return self.next_type_ouput.forward(x), self.mask_output.forward(x)
