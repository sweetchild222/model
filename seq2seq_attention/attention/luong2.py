import tensorflow as tf


class Luong2(tf.keras.layers.Layer):

  def __init__(self, units):
    super(Luong2, self).__init__()

    self.w = tf.keras.layers.Dense(units)
    

  def call(self, query, values):

    query = tf.expand_dims(query, -1)

    score = self.w(tf.matmul(values, query))

    score = tf.reduce_sum(score, axis=2, keepdims=True)

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

