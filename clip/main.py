#torch version : 2.0.0+cu118
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from data_loader.custom_dataset import CustomDataset
from data_loader.tokenizer import Tokenizer
from data_loader.image_tensor_converter import ImageTensorConverter

import numpy as np


images_path = ['images/astronaut.png',
          'images/camera.png',
          'images/chelsea.png',
          'images/coffee.png',
          'images/horse.png',
          'images/motorcycle_left.png',
          'images/page.png',
          'images/rocket.jpg']

texts = ['a portrait of an astronaut with the American flag!',
          'a person looking at a camera on a tripod',
          'a facial photo of a tabby cat',
          'a cup of coffee on a saucer',
          'a black-and-white silhouette of a horse',
          'a red motorcycle standing in a garage',
          'a page of text about segmentatio',
          'a rocket standing on a launchpad']



def create_image_encoder(input_resolution, out_dim):

    kernel_size = 8
    hidden = 256
    transformer_count = 3
    attn_head_count = hidden // 64

    encoder = ImageEncoder(input_resolution, kernel_size, hidden, transformer_count, attn_head_count, out_dim)

    return encoder.train()


def create_text_encoder(seq_len, vocab_size, out_dim):

    hidden = 128
    transformer_count = 2
    attn_head_count = 4

    encoder = TextEncoder(seq_len, vocab_size, hidden, transformer_count, attn_head_count, out_dim)
    
    return encoder.train()


def create_data_loader(seq_len, image_tensor_resolution):

    batch_size = 3 
    image_tensor_converter = ImageTensorConverter(image_tensor_resolution)

    return DataLoader(CustomDataset(images_path, image_tensor_converter, texts, tokenizer, seq_len), batch_size=batch_size)


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer(texts)

seq_len = 15
out_dim = 256
image_tensor_resolution = 224

image_encoder = create_image_encoder(image_tensor_resolution, out_dim).to(device)
text_encoder = create_text_encoder(seq_len, tokenizer.vocab_size(), out_dim).to(device)
data_loader = create_data_loader(seq_len, image_tensor_resolution)

optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()))
logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


def forward(image, text):

    image_predict = image_encoder.forward(image)
    text_predict = text_encoder.forward(text)

    image_predict = image_predict / image_predict.norm(dim=-1, keepdim=True)
    text_predict = text_predict / text_predict.norm(dim=-1, keepdim=True)

    #cosine similarity as logits
    predict = logit_scale.exp() * torch.matmul(image_predict, text_predict.t())

    return nn.LogSoftmax(dim=-1).forward(predict)    


def backward(loss):

    optimizer.zero_grad()

    loss.backward()
    
    optimizer.step()


def train():

    epochs = 10000

    for epoch in range(epochs):

        correct = 0
        total = 0
        loss_total = 0

        for i, (image, text) in enumerate(data_loader):

            predict = forward(image.to(device), text.to(device))
            
            target = torch.arange(image.shape[0], dtype=torch.long, device=device)

            loss = nn.NLLLoss().forward(predict, target)

            backward(loss)

            loss_total += loss
            arg_max = predict.argmax(dim=-1)
            correct += arg_max.eq(target).sum().item()
            total += arg_max.shape[0]


        print("epoch=" + str(epoch + 1) + "/" + str(epochs) + ", "
                "correct=" + str(correct) + '/' + str(total) + "(" + str(correct * 100 // total) + "%), "
                "loss=" + str(loss_total.item() / (i + 1)))

train()