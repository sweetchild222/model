import torch
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    
    def __init__(self, images_path, image_tensor_converter, texts, tokenizer, seq_len):
        
        self.images_path = images_path
        self.image_tensor_converter = image_tensor_converter

        self.image_tensor_dir = 'image_tensor'
        self.create_image_tensor_dir(dir=self.image_tensor_dir)

        self.texts  = texts        
        self.tokenizer = tokenizer
        self.seq_len = seq_len


    def create_image_tensor_dir(self, dir):

        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            for file in os.listdir(dir):
                try:
                    os.unlink(dir + '/' + file)
                except Exception as e:
                    print(e)
                    exit()


    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, index):

        text_tensor = self.text_to_tensor(self.texts[index])

        image_tensor = self.image_to_tensor(self.images_path[index])

        return image_tensor, text_tensor
    

    def image_to_tensor(self, image_path):

        tensor_path = self.image_tensor_path(image_path)

        if os.path.exists(tensor_path):
            return torch.load(tensor_path, weights_only=True)
        
        tensor = self.image_tensor_converter.covert(image_path)

        torch.save(tensor, tensor_path)
    
        return tensor
    
   
    def image_tensor_path(self, image_path):

        filename = os.path.basename(image_path)

        return self.image_tensor_dir + '/' + filename
    

    def text_to_tensor(self, text):

        token = self.tokenizer.encode(text)

        if len(token) > self.seq_len:
            raise RuntimeError(f"{text} is too long for sequence length {self.seq_len}")
    
        tensor = torch.zeros(self.seq_len, dtype=torch.int)

        tensor[:len(token)] = torch.tensor(token)        

        return tensor


