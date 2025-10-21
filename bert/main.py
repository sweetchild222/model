#torch version : 2.0.0+cu118
import torch
from torch.utils.data import DataLoader

from model.bert import BERT
from bert_dataset import BERTDataset



device = "cuda" if torch.cuda.is_available() else "cpu"

file_path = 'small.csv'

bert_dataset = BERTDataset(file_path, seq_len=30)

print('-----------------------')
print('file: ', file_path)
print('vocab size: ', bert_dataset.vocab_size())
print('dateset length: ', len(bert_dataset))
print('-----------------------')

bert = BERT(vocab_size=bert_dataset.vocab_size(), embed_size=256, encoder_count=2, attention_head_count=4).to(device)

next_type_loss_layer = torch.nn.NLLLoss()
mask_loss_layer = torch.nn.NLLLoss(ignore_index=bert_dataset.pad_index)

adam = torch.optim.AdamW(bert.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)


def forward(x, segment, next_type_target, mask_target):

    mask = (x != bert_dataset.pad_index).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    
    next_type_predic, mask_predic = bert.forward(x, segment, mask)
    
    next_type_loss = next_type_loss_layer.forward(next_type_predic, next_type_target)
    
    mask_loss = mask_loss_layer.forward(mask_predic.transpose(1, 2), mask_target)

    return next_type_predic, mask_predic, next_type_loss, mask_loss


def backward(next_type_loss, mask_loss):

    adam.zero_grad()

    (next_type_loss + mask_loss).backward()
    
    adam.step()


def print_sentence(word_input, next_type_target, next_type_predic, mask_target, mask_predic, mask_filter):

    print()
    print('word input tensor: ', word_input.tolist())
    print('word input sentence: ', bert_dataset.sentence(word_input))
    print()
    
    print('next type target: ', next_type_target.item(), '(' + bert_dataset.next_type_label(next_type_target.item()) +')')
    print('next type predic: ', next_type_predic.item(), '(' + bert_dataset.next_type_label(next_type_predic.item()) +')')
    print()
    
    print('mask target tensor: ', mask_target.tolist())
    print('mask target sentence: ', bert_dataset.sentence(mask_target))
    print()

    filterd_predic = mask_predic.tolist()
    
    for i, filter in enumerate(mask_filter):
        if filter == False:
            filterd_predic[i] = bert_dataset.pad_index

    print('mask predic tensor: ', filterd_predic)
    print('mask predic sentence: ', bert_dataset.sentence(filterd_predic))
    print()


def main():

    epochs = 6000

    data_loader = DataLoader(bert_dataset, batch_size=512, shuffle=True, num_workers=5)

    for epoch in range(epochs):

        next_type_correct = 0
        next_type_total = 0
        next_type_loss_total = 0	    
        mask_correct = 0
        mask_total = 0
        mask_loss_total = 0	    

        for i, (word_input, segment_input, next_type_target, mask_target) in enumerate(data_loader):

            word_input = word_input.to(device)
            segment_input = segment_input.to(device)
            next_type_target = next_type_target.to(device)
            mask_target = mask_target.to(device)

            next_type_predic, mask_predic, next_type_loss, mask_loss = forward(word_input, segment_input, next_type_target, mask_target)

            backward(next_type_loss, mask_loss)

            mask_loss_total += mask_loss
            mask_filter = (mask_target != bert_dataset.pad_index)
            mask_predic = mask_predic.argmax(dim=-1)
            mask_correct += mask_predic[mask_filter].eq(mask_target[mask_filter]).sum().item()
            mask_total += torch.sum(mask_filter == True).item()

            next_type_loss_total += next_type_loss
            next_type_predic = next_type_predic.argmax(dim=-1)
            next_type_correct += next_type_predic.eq(next_type_target).sum().item()
            next_type_total += next_type_predic.shape[0]

            if len(data_loader) == (i + 1) and epoch % 99 == 0:
                print_sentence(word_input[0], next_type_target[0], next_type_predic[0], mask_target[0], mask_predic[0], mask_filter[0])

        print("epoch="  + str(epoch + 1) + "/" + str(epochs) + ", "
            "next_type_correct=" + str(next_type_correct) + "/" + str(next_type_total) + "(" + str(next_type_correct * 100 // next_type_total) + "%), "
            "next_type_loss=" + str(next_type_loss_total.item() / (i + 1)) + ", "
            "mask_correct=" + str(mask_correct) + "/" + str(mask_total) + "(" + str(mask_correct * 100 // mask_total) + "%), "
            "mask_loss=" + str(mask_loss_total.item() / (i + 1)))


main()
