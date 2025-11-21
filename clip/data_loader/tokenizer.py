from .byte_pair_encoding import BytePairEncoding
from .common import *


class Tokenizer(object):
    def __init__(self, texts):
        
        self.bpe = BytePairEncoding(texts)
        
        vocabulary = set(bytes_to_unicode().values())
        vocabulary.update({v + end_tag_word() for v in vocabulary})
        vocabulary.update({start_tag_sentence(), end_tag_sentence(), end_tag_word()})
        vocabulary.update({''.join(b) for b in self.bpe.vocabulary()})

        self.word2index = dict(zip(vocabulary, range(len(vocabulary))))
        self.index2word = {v: k for k, v in self.word2index.items()}    


    def vocab_size(self):

        return len(self.word2index)


    def decode(self, tokens):

        sentence = ''

        for token in tokens:
            sentence += self.index2word[token] + ' '

        return sentence


    def encode(self, text):
        
        word_list = self.bpe.encode(text)

        tokens = [self.word2index[start_tag_sentence()]]

        for word in word_list:
            tokens.append(self.word2index[word])

        tokens.append(self.word2index[end_tag_sentence()])

        return tokens                
