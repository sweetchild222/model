import pandas as pd
import re


def preprocess(sentence):

  sentence = re.sub(r"([?.!,])", r" \1 ", sentence) # 12시 땡! -> 12시 땡 !

  return sentence.strip()


def load_dataset(path):

  train_data = pd.read_csv(path)

  questions = [preprocess(sentence) for sentence in train_data['Q']]

  answers = [preprocess(sentence) for sentence in train_data['A']]

  return questions, answers
