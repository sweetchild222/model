import torch
import torch.nn.functional as F
import json

from torch.optim import AdamW
from model.clip import CLIP
from data_loader.CLIP_dataset import CLIPDataset
from data_loader.cosin_scheduler import cosin_scheduler
from data_loader.tokenizer import Tokenizer
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():

    annotation_path = 'clip_data/annotation.json'
    image_folder_path = 'clip_data/image'

    with open(annotation_path, 'r', encoding='utf-8') as file:

        annotations = json.load(file)

        ids = list(annotations.keys())

        texts = []
        images = []

        for id in ids:
            texts.append(annotations.get(id))
            image_path = image_folder_path + '/' + str(id).zfill(12) + '.jpg'
            images.append(image_path)

        return texts, images


def create_tokenizer(texts):

    return Tokenizer([t for text in texts for t in text])


def create_dataloader(texts, tokenizer, images, img_resolution):
            
    dataset = CLIPDataset(texts, images, tokenizer, context_length, img_resolution)

    return DataLoader(dataset, shuffle=True, batch_size=64, num_workers=4)
    


def create_model(vocab_size, context_length, img_resolution):

    params = {'embed_dim': 1024,
              'img_resolution': img_resolution,
              'img_width': 64,
              'context_length': context_length,
              'vocab_size': vocab_size,
              'trans_d_model': 512,
              'trans_heads': 8,
              'trans_layers': 3}
    

    model = CLIP(**params)
    model = model.to(device)

    return model
    

context_length = 70
img_resolution = 160

texts, images = load_data()

tokenizer = create_tokenizer(texts)

dataloader = create_dataloader(texts, tokenizer, images, img_resolution)

model = create_model(tokenizer.vocab_size(), context_length, img_resolution)


def forward(input_images, input_texts, target):
    
    image_features, text_features = model(input_images, input_texts)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
    logit_scale = model.logit_scale.exp()

    image_predict = logit_scale * image_features @ text_features.t()
    text_predict = logit_scale * text_features @ image_features.t()

    image_loss = F.cross_entropy(image_predict, target)
    text_loss  = F.cross_entropy(text_predict, target)

    loss = (image_loss + text_loss) / 2

    return image_predict, text_predict, loss


def backward(loss, optimizer, scheduler):

    optimizer.zero_grad()

    loss.backward()    

    optimizer.step()
    
    scheduler.step()


def main():

    epochs = 50

    total_step = len(dataloader) * epochs

    optimizer = AdamW(model.parameters(), lr=5e-4, eps=1.0e-08, weight_decay=0.1)
    
    scheduler = cosin_scheduler(optimizer, warmup_steps=total_step // 5, training_steps=total_step)        

    for epoch in range(epochs):

        total_loss = 0
        image_correct = 0
        text_correct = 0
        total_count = 0
        
        for i, (input_images, input_texts) in enumerate(dataloader):
                        
            input_images = input_images.to(device)
            input_texts = input_texts.to(device)

            target = torch.arange(len(input_images)).to(device)

            image_predict, text_predict, loss = forward(input_images, input_texts, target)

            image_correct += image_predict.argmax(dim=-1).eq(target).sum().item()
            text_correct += text_predict.argmax(dim=-1).eq(target).sum().item()

            total_loss += loss.item()

            total_count += target.shape[0]
            
            backward(loss, optimizer, scheduler)

        print("epoch="  + str(epoch + 1) + "/" + str(epochs) + ", "
            "image_correct=" + str(image_correct) + "/" + str(total_count) + "(" + str(image_correct * 100 // total_count) + "%), "
            "text_correct=" + str(text_correct) + "/" + str(total_count) + "(" + str(text_correct * 100 // total_count) + "%), "
            "loss=" + str(total_loss / (i + 1)))        

main()
