import os
import numpy as np

from discriminator import Discriminator
from generator import Generator

#torch version : 2.0.0+cu118
import torch
from torchvision import *
from torch.autograd import Variable


def load(img_size):

    global datasets

    download_path = "./"
    os.makedirs(download_path, exist_ok=True)
    datasets = datasets.MNIST(download_path, train=True, download=True, transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))

    return torch.utils.data.DataLoader(datasets, batch_size=64, shuffle=True)


channel = 1
img_size=28
latent_dim=100 
input_shape = (channel, img_size, img_size)

adversarial_loss = torch.nn.BCELoss()
generator = Generator(input_shape, latent_dim)
discriminator = Discriminator(input_shape)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

dataloader = load(img_size)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

epochs = 50

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        valid = Variable(tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        invalid = Variable(tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
                
        real = Variable(imgs.type(tensor))

        optimizer_G.zero_grad()

        noise = Variable(tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        fake = generator(noise)

        g_loss = adversarial_loss(discriminator(fake), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real), valid)
        fake_loss = adversarial_loss(discriminator(fake.detach()), invalid)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if len(dataloader) == (i + 1):

            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % ((epoch + 1), epochs, d_loss.item(), g_loss.item()))

            fake_path = "fake"
            os.makedirs(fake_path, exist_ok=True)
            utils.save_image(fake.data[:25], (fake_path + "/%d.png" % epoch), nrow=5, normalize=True)