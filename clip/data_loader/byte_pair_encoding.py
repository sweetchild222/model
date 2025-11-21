import re, collections
from .common import *


class BytePairEncoding:

    def __init__(self, texts):

        self.byte_encoder = bytes_to_unicode()

        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        self.pat = re.compile(pattern, re.IGNORECASE)

        word_count = self.make_word_count(texts)

        self.bpe_codes = self.make_bpe(word_count)


    def vocabulary(self):

        return self.bpe_codes.keys()
    

    def make_word_count(self, texts):

        dict = {}
        for text in texts:
            text = text_clean(text)
            
            for word in re.findall(self.pat, text):

                word = ' '.join(self.byte_encoder[b] for b in word.encode('utf-8'))
                word += ' ' + end_tag_word()

                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 0

        return dict


    def merge(self, pair, word_count):

        word_count_out = {}
        bigram = re.escape(' '.join(pair))
        #?<! = Negative lookbehind
        #?! = Negative lookahead
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in word_count:
            word_out = p.sub(''.join(pair), word)
            word_count_out[word_out] = word_count[word]
            
        return word_count_out


    def get_stats(self, word_count):

        pairs = collections.defaultdict(int)
    
        for word, freq in word_count.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq

        return pairs


    def make_bpe(self, word_count):
        
        #min_frequency = 50 # 19 % words splited
        #min_frequency = 20 # 12 % words splited
        #min_frequency = 10 # 8.2 % words splited
        min_frequency = 2 # 2.9 % words splited
        #min_frequency = 0 # 0 % words splited

        bpe_codes = {}
        i = 0

        while True:
        
            pairs = self.get_stats(word_count)

            if len(pairs) == 0:
                break
        
            max_pair = max(pairs, key=pairs.get)

            if pairs[max_pair] < min_frequency:
                break
        
            word_count = self.merge(max_pair, word_count)

            bpe_codes[max_pair] = i

            i += 1

        return bpe_codes
    

    def get_pairs(self, word):

        pairs = set()
        prev_char = word[0]

        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char

        return pairs


    def encode(self, text):
        
        text = text_clean(text)

        word_list = []

        for word in re.findall(self.pat, text):

            word = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))
            
            word_list.extend([w for w in self.encode_core(word)])
        
        return word_list


    def encode_core(self, ori_word):

        word = tuple(ori_word) + (end_tag_word(),)
        pairs = self.get_pairs(word)

        if not pairs:   #if ori_word empty
            return ori_word

        while True:        
            bigram = min(pairs, key = lambda pair: self.bpe_codes.get(pair, float('inf')))        
            if bigram not in self.bpe_codes:
                break
            
            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        return word
