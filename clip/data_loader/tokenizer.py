import re

class Tokenizer:

    def __init__(self, texts):

        self.sos_index = 1
        self.eos_index = 2
        
        self.special_words = {self.sos_index: "<S>", self.eos_index: "<E>"}

        self.texts = [self.detach_punctuation(text) for text in texts]

        self.index2word, self.word2index = self.make_word_index(list(self.special_words.values()), self.texts)

        
    def vocab_size(self):

        return len(self.word2index)
    

    def detach_punctuation(self, text):
            
        return re.sub(r"([?.!,])", r" \1", text.strip())

    
    def convert_index2word(self, list):

        str = ''

        for i in list:
            str += self.word(i) + ' '

        return str


    def make_word_index(self, special_words, texts):

        word_set = set()
        
        for text in texts:
            text = text.split()
            
            for word in text:
                word_set.add(word)
            
        index2word = special_words + list(word_set)
        word2index = {token: index for index, token in enumerate(index2word)}

        return index2word, word2index
    

    def encode(self, text):

        text = self.detach_punctuation(text)

        tokens = [self.sos_index]
        
        for word in text.split():

            index = self.word2index[word]

            tokens.append(index)

        tokens.append(self.eos_index)

        return tokens




        





