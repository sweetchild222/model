import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_datasets as tfds

from transformer import *
from loader import *


def custom_loss(max_length):

  def loss(y_true, y_pred):

    y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)
  
  return loss


class custom_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(custom_schedule, self).__init__()
    
    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):

    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def custom_accuracy(max_length):

  def accuracy(y_true, y_pred):
  
    # (batch_size, max_length - 1)
    y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
  
  return accuracy


def max_length(sentences):

  max_length = 0
  
  for sentence in sentences:

    length = len(sentence)

    if(length > max_length):
      max_length = length

  return max_length


def tokenize_and_padding(sentences, start_token, end_token):

  # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
  tokenized = [start_token + tokenizer.encode(sentence) + end_token for sentence in sentences]  

  return tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=max_length(tokenized), padding='post')



questions, answers = load_dataset('data.csv')

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

tokenized_string = tokenizer.encode(questions[20])
original_string = tokenizer.decode(tokenized_string)
print(tokenized_string, ' - ', original_string)

start_token = [tokenizer.vocab_size]
end_token = [tokenizer.vocab_size + 1]
vocab_size = tokenizer.vocab_size + 2 #for adding start token and end token

questions = tokenize_and_padding(questions, start_token, end_token)
answers = tokenize_and_padding(answers, start_token, end_token)
output_max_length = answers.shape[-1]

print('questions.shape:', questions.shape)
print('answers.shape :', answers.shape)
print('vocab size:', vocab_size)

dataset = tf.data.Dataset.from_tensor_slices(({'enc_inputs': questions,'dec_inputs': answers[:, :-1]}, {'outputs': answers[:, 1:]}))
dataset = dataset.cache()
dataset = dataset.shuffle(20000)
dataset = dataset.batch(64)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()

d_model = 256

optimizer = tf.keras.optimizers.Adam(custom_schedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model = transformer(vocab_size=vocab_size, num_layers=2, dff=512, d_model=d_model, num_heads=8, dropout=0.1)
model.compile(optimizer=optimizer, loss=custom_loss(output_max_length), metrics=[custom_accuracy(output_max_length)])
model.fit(dataset, epochs=50)


def evaluate(sentence, start_token, end_token, output_max_length):

  sentence = preprocess(sentence)

  enc_input = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

  dec_input = tf.expand_dims(start_token, 0)

  for i in range(output_max_length):

    predictions = model(inputs=[enc_input, dec_input], training=False)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    if tf.equal(predicted_id, end_token[0]):
      break
    
    dec_input = tf.concat([dec_input, predicted_id], axis=-1)

  return tf.squeeze(dec_input, axis=0)


def predict(sentence, start_token, end_token, output_max_length):

  prediction = evaluate(sentence, start_token, end_token, output_max_length)

  predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])  

  return predicted_sentence




test_questions = ['영화 볼래?', '고민이 있어', '너무 화가나', '게임하고싶은데 할래?', '나 너 좋아하는 것 같아', '딥 러닝 자연어 처리를 잘 하고 싶어']

for test_question in test_questions:

  test_answer = predict(test_question, start_token, end_token , output_max_length)
  
  print(test_question, ' --> ', test_answer)

