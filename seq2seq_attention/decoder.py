import tensorflow as tf


class Decoder(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, units, batch_size, attention):
    super(Decoder, self).__init__()

    self.batch_size = batch_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = attention


  def call(self, x, hidden, output):

    context_vector, attention_weights = self.attention(hidden, output)

    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    return x, state, attention_weights

