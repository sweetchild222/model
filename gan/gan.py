import os

from discriminator import Discriminator
from generator import Generator

#torch version : 2.0.0+cu118
import torch
from torchvision import *


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

adversarial_loss = torch.nn.BCELoss().to(device)
generator = Generator(input_shape, latent_dim).to(device)
discriminator = Discriminator(input_shape).to(device)

dataloader = load(img_size)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        valid = torch.ones(size=(imgs.size(0), 1), dtype=torch.float32, requires_grad=False).to(device)
        invalid = torch.zeros(size=(imgs.size(0), 1), dtype=torch.float32, requires_grad=False).to(device)
        real = imgs.to(device)

        optimizer_G.zero_grad()

        noise = torch.normal(mean=0.0, std=1.0, size=(imgs.shape[0], latent_dim), dtype=torch.float32).to(device)

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