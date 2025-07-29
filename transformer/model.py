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
  
  tokenized = [start_token + tokenizer.encode(sentence) + end_token for sentence in sentences]

  return tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=max_length(tokenized), padding='post')


def create_model(vocab_size, num_layers, dff, d_model, num_heads, output_max_length):
  
  optimizer = tf.keras.optimizers.Adam(custom_schedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  model = transformer(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=0.1)
  model.compile(optimizer=optimizer, loss=custom_loss(output_max_length), metrics=[custom_accuracy(output_max_length)])

  return model
