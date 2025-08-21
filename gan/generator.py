import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Generator, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.input_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        fake = self.model(z)
        fake = fake.view(fake.size(0), *self.input_shape)
        return fake