import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()

    self.w1 = tf.keras.layers.Dense(units)
    self.w2 = tf.keras.layers.Dense(units)
    self.v = tf.keras.layers.Dense(1)

  def call(self, query, values):

    query_with_time_axis = tf.expand_dims(query, 1)

    score = self.v(tf.nn.tanh(self.w1(query_with_time_axis) + self.w2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

