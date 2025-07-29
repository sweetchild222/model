import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_datasets as tfds

from create_model import *
from loader import *


def tokenize_and_padding(tokenizer, sentences, start_token, end_token):

  tokenized_list = [start_token + tokenizer.encode(sentence) + end_token for sentence in sentences]

  max_length = max([len(tokenized) for tokenized in tokenized_list])

  return tf.keras.preprocessing.sequence.pad_sequences(tokenized_list, maxlen=max_length, padding='post')


def create_dataset(questions, answers, shuffle, batch):
  
  dataset = tf.data.Dataset.from_tensor_slices(({'enc_inputs': questions,'dec_inputs': answers[:, :-1]}, {'outputs': answers[:, 1:]}))
  dataset = dataset.cache()
  dataset = dataset.shuffle(shuffle)
  dataset = dataset.batch(batch)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


questions, answers = load_dataset('data.csv')

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

start_token = [tokenizer.vocab_size]
end_token = [tokenizer.vocab_size + 1]
vocab_size = tokenizer.vocab_size + 2 #adding 2 for start token and end token

questions = tokenize_and_padding(tokenizer, questions, start_token, end_token)
answers = tokenize_and_padding(tokenizer, answers, start_token, end_token)
output_max_length = answers.shape[-1]

print('questions.shape:', questions.shape)
print('answers.shape :', answers.shape)
print('vocab size:', vocab_size)

model = create_model(vocab_size=vocab_size, num_layers=2, dff=512, d_model=256, num_heads=8, output_max_length=output_max_length)
model.fit(create_dataset(questions, answers, 20000, 64), epochs=50)


def predict(sentence):

  enc_input = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

  dec_input = tf.expand_dims(start_token, 0)

  for i in range(output_max_length):

    predictions = model(inputs=[enc_input, dec_input], training=False)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    if tf.equal(predicted_id, end_token[0]):
      break
    
    dec_input = tf.concat([dec_input, predicted_id], axis=-1)

  prediction = tf.squeeze(dec_input, axis=0)

  return tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])


test_questions = ['영화 볼래?', '고민이 있어', '너무 화가나', '게임하고싶은데 할래?', '나 너 좋아하는 것 같아', '딥 러닝 자연어 처리를 잘 하고 싶어']
test_answers = [predict(preprocess(test_question)) for test_question in test_questions]

print(test_questions)
print(test_answers)

