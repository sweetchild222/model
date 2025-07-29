import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import re

from encoder import *
from decoder import *


def create_padding_mask(x):

  mask = tf.cast(tf.math.equal(x, 0), tf.float32)

  #(batch_size, 1, 1, key sequence)
  return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):

  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)

  return tf.maximum(look_ahead_mask, padding_mask)


def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):

  enc_inputs = tf.keras.Input(shape=(None,), name="enc_inputs")

  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(enc_inputs)

  look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)

  dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(enc_inputs)

  enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,)(
                        inputs=[enc_inputs, enc_padding_mask])
  
  dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,)(
                        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
  
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs, name=name)


max_length = 40

def loss_function(y_true, y_pred):

  y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):

    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


train_data = pd.read_csv('data.csv')
train_data.head()

print('sample count :', len(train_data))

print(train_data.isnull().sum())

questions = []

for sentence in train_data['Q']:    
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []

for sentence in train_data['A']:    
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

len(questions)

print(questions[:5])
print(answers[:5])

# 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2

print('시작 토큰 번호 :',START_TOKEN)
print('종료 토큰 번호 :',END_TOKEN)
print('단어 집합의 크기 :',VOCAB_SIZE)

# 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환.
print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

sample_string = questions[20]

tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))


for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

def tokenize_and_filter(inputs, outputs):

  tokenized_inputs = []
  tokenized_outputs = []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)
  
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=max_length, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=max_length, padding='post')
  
  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)


print('질문 데이터의 크기(shape) :', questions.shape)
print('답변 데이터의 크기(shape) :', answers.shape)
print(questions[0])
print(answers[0])
print('단어 집합의 크기(Vocab size): {}'.format(VOCAB_SIZE))
print('전체 샘플의 수(Number of samples): {}'.format(len(questions)))


# 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다., dec_inputs 디코더의 입력. 마지막 패딩 토큰이 제거된다.  answer 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
dataset = tf.data.Dataset.from_tensor_slices(({'enc_inputs': questions,'dec_inputs': answers[:, :-1]}, {'outputs': answers[:, 1:]}))
dataset = dataset.cache()
dataset = dataset.shuffle(20000)
dataset = dataset.batch(64)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(answers[0])
print(answers[:1][:, :-1])
print(answers[:1][:, 1:])

tf.keras.backend.clear_session()

d_model = 256

model = transformer(vocab_size=VOCAB_SIZE, num_layers=2, dff=512, d_model=d_model, num_heads=8, dropout=0.1)
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)



def accuracy(y_true, y_pred):
  
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, max_length - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.fit(dataset, epochs=50)

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 예측 시작
  for i in range(max_length):
    predictions = model(inputs=[sentence, output], training=False)

    # 현재(마지막) 시점의 예측 단어를 받아온다.
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 마지막 시점의 예측 단어를 출력에 연결한다.
    # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence


output = predict('영화 볼래?')
output = predict("고민이 있어")
output = predict("너무 화가나")
output = predict("게임하고싶은데 할래?")
output = predict("나 너 좋아하는 것 같아")
output = predict("딥 러닝 자연어 처리를 잘 하고 싶어")
