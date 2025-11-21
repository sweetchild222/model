import torch
from torch.utils.data import Dataset

from PIL import Image
from .img_transform import img_transform
import random

class CLIPDataset(Dataset):

    def __init__(self, texts, images, tokenizer, context_length, img_resolution):
        super().__init__()
        
        self.texts = texts
        self.images = images

        self.tokenizer = tokenizer
        self.transform = img_transform(img_resolution)
        self.context_length = context_length


    def vocab_size(self):
        
        return self.tokenizer.vocab_size()
            

    def __len__(self):
        
        return len(self.texts)


    def __getitem__(self, idx):
        
        img = Image.open(self.images[idx])
        img_input = self.transform(img)

        text = random.choice(self.texts[idx])
        tokens = self.tokenizer.encode(text)
        
        text_input = torch.zeros(self.context_length, dtype=torch.long)
        text_input[:len(tokens)] = torch.tensor(tokens)

        return img_input, text_input
