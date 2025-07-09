import unicodedata
import re
import io


def unicode_to_ascii(s):

  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, head, tail):

  w = unicode_to_ascii(w.lower().strip())

  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()

  return head + ' ' + w + ' ' + tail


def create_dataset(path, num_examples, head, tail):

  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w, head, tail) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)
