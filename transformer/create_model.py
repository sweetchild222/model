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


def create_model(vocab_size, num_layers, dff, d_model, num_heads, output_max_length):
  
  optimizer = tf.keras.optimizers.Adam(custom_schedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  model = transformer(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=0.1)
  model.compile(optimizer=optimizer, loss=custom_loss(output_max_length), metrics=[custom_accuracy(output_max_length)])

  return model
