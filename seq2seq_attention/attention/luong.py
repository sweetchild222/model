import tensorflow as tf


class Luong(tf.keras.layers.Layer):

  def __init__(self):
    super(Luong, self).__init__()


  def call(self, query, values):

    query = tf.expand_dims(query, -1)

    score = tf.matmul(values, query)

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
