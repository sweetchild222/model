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


def max_length_calc(sentences):

  max_length = 0
  
  for sentence in sentences:

    length = len(tokenizer.encode(sentence))

    if(length > max_length):
      max_length = length

  return max_length



questions, answers = create_dataset('data.csv')

max_length = 40
print(questions[:5])
print(answers[:5])

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

START_TOKEN = [tokenizer.vocab_size]
END_TOKEN = [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2 #for adding start token and end token

# 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환.
print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

sample_string = questions[20]

tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))


for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))


def tokenize_and_padding(inputs, outputs, start_token, end_token):

  tokenized_inputs = []
  tokenized_outputs = []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
    sentence1 = start_token + tokenizer.encode(sentence1) + end_token
    sentence2 = start_token + tokenizer.encode(sentence2) + end_token  

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

  input_max_length = max_length_calc(tokenized_inputs)
  output_max_length = max_length_calc(tokenized_outputs)
  
  print(input_max_length, output_max_length)

  exit()

  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=max_length, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=max_length, padding='post')
  
  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_padding(questions, answers, START_TOKEN, END_TOKEN)



print('질문 데이터의 크기(shape) :', questions.shape)
print('답변 데이터의 크기(shape) :', answers.shape)
print('단어 집합의 크기(Vocab size): {}'.format(VOCAB_SIZE))


# 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다., dec_inputs 디코더의 입력. 마지막 패딩 토큰이 제거된다.  answer 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
dataset = tf.data.Dataset.from_tensor_slices(({'enc_inputs': questions,'dec_inputs': answers[:, :-1]}, {'outputs': answers[:, 1:]}))
dataset = dataset.cache()
dataset = dataset.shuffle(20000)
dataset = dataset.batch(64)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()

d_model = 256
optimizer = tf.keras.optimizers.Adam(custom_schedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model = transformer(vocab_size=VOCAB_SIZE, num_layers=2, dff=512, d_model=d_model, num_heads=8, dropout=0.1)
model.compile(optimizer=optimizer, loss=custom_loss(max_length), metrics=[custom_accuracy(max_length)])
model.fit(dataset, epochs=50)


def evaluate(sentence, start_token, end_token):

  sentence = preprocess(sentence)

  sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

  output = tf.expand_dims(start_token, 0)

  for i in range(max_length):
    
    predictions = model(inputs=[sentence, output], training=False)

    # 현재(마지막) 시점의 예측 단어를 받아온다.
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
    if tf.equal(predicted_id, end_token[0]):
      break

    # 마지막 시점의 예측 단어를 출력에 연결한다.
    # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence, start_token, end_token):

  prediction = evaluate(sentence, start_token, end_token)

  predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])  

  return predicted_sentence


test_questions = ['영화 볼래?', '고민이 있어', '너무 화가나', '게임하고싶은데 할래?', '나 너 좋아하는 것 같아', '딥 러닝 자연어 처리를 잘 하고 싶어']

for test_question in test_questions:

    test_answer = predict(test_question, START_TOKEN, END_TOKEN)

    print(test_question, ' --> ', test_answer)

