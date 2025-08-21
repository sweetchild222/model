import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):

        input_flat = input.view(input.size(0), -1)

        return self.model(input_flat)        
