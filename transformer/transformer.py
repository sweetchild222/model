import tensorflow as tf

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

  enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(
                        inputs=[enc_inputs, enc_padding_mask])
  
  dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(
                        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
  
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs, name=name)


