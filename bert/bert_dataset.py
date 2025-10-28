import torch
import random
import re
from torch.utils.data import Dataset


class BERTDataset(Dataset):

    def __init__(self, path, seq_len):
        
        self.seq_len = seq_len
        
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2
        self.eos_index = 3
        self.mask_index = 4

        self.special_words = {self.pad_index: "<P>", self.unk_index:"<U>", self.sos_index:"<S>", self.eos_index: "<E>", self.mask_index: "<M>"}
        
        self.texts = self.trim_lines(self.load(path))

        self.index2word, self.word2index = self.make_word_index(list(self.special_words.values()), self.texts)


    def detach_punctuation(self, text):
            
        # 12시 땡! -> 12시 땡 !
        return re.sub(r"([?.!,])", r" \1", text.strip())


    def trim_lines(self, lines):

        new_texts = []

        for line in lines:
            
            texts = []
                        
            texts.append(self.detach_punctuation(line[0].strip()))
            texts.append(self.detach_punctuation(line[1].strip()))
            texts.append(line[2].strip())

            new_texts.append(texts)

        return new_texts
    

    def make_word_index(self, special_words, texts):

        word_set = set()

        for text in texts:
            for t in text[0:2]:
                
                words = t.split()

                for word in words:
                    word_set.add(word)

        index2word = special_words + list(word_set)
        word2index = {token: index for index, token in enumerate(index2word)}

        return index2word, word2index


    def load(self, path):
        
        with open(path, "r", encoding='utf-8') as file:
            return [line[:-1].split(",") for line in file]
        
        
    def sentence(self, index_list):

        str = ''

        for index in index_list:
            str += self.index2word[index] + ' '

        return str


    def word(self, index):

        return self.index2word[index]

    
    def index(self, word):

        return self.word2index.get(word, self.unk_index)


    def vocab_size(self):

        return len(self.word2index)


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        t1_sentence, t2_sentence, next_type_target = self.random_sentence(index)

        t1_input, t1_target = self.random_word(t1_sentence.split())
        t2_input, t2_target = self.random_word(t2_sentence.split())
        
        t1_input = [self.sos_index] + t1_input + [self.eos_index]
        t1_target = [self.pad_index] + t1_target + [self.pad_index]

        t2_input = t2_input + [self.eos_index]
        t2_target = t2_target + [self.pad_index]
        
        word_input = (t1_input + t2_input)[:self.seq_len]
        mask_target = (t1_target + t2_target)[:self.seq_len]
        
        segment_input = ([1] * len(t1_input) + [2] * len(t2_input))[:self.seq_len]
        
        padding = [self.pad_index] * (self.seq_len - len(word_input))
        
        word_input.extend(padding)
        mask_target.extend(padding)
        segment_input.extend(padding)

        return torch.tensor(word_input), torch.tensor(segment_input), torch.tensor(next_type_target), torch.tensor(mask_target)


    def convert_index2word(self, list):

        str = ''

        for i in list:
            str += self.word(i) + ' '

        return str

    def random_word(self, words):
    
        input = []
        target = []

        for word in words:            
            prob = random.random()

            if prob < 0.15:
                prob /= 0.15
                
                if prob < 0.8:      # 80% randomly change token to mask token
                    input.append(self.mask_index)
                elif prob < 0.9:    # 10% randomly change token to random token
                    input.append(random.randrange(len(self.special_words), self.vocab_size()))
                else:               # 10% randomly change token to current token
                    input.append(self.index(word))

                target.append(self.index(word))

            else:
                input.append(self.index(word))
                target.append(0)
    
        return input, target


    def next_type_label(self, index):
        
        labels = ['ordinary', 'negative', 'positive', 'mismatch']
        
        return labels[index]


    def next_type_index(self, label):

        indices = {'ordinary':0 , 'negative': 1, 'positive':2, 'mismatch':3 }

        return indices[label]

        
    def random_sentence(self, index):

        text = self.texts[index]
        
        r = random.random()
        
        if r < 0.75:
            return text[0], text[1], int(text[2])
        else:
            return text[0], self.get_random_text(index), self.next_type_index('mismatch')


    def get_random_text(self, exclude_index):
        
        find_index = -1

        while True:
            find_index = random.randrange(len(self.texts))

            if exclude_index != find_index:
                return self.texts[find_index][1]
