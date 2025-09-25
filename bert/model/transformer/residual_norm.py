import torch.nn as nn


class ResidualNorm(nn.Module):

    def __init__(self, size):
        super(ResidualNorm, self).__init__()

        self.norm = nn.LayerNorm(size)

    def forward(self, x):

        return x + self.norm(x) #residual connection
