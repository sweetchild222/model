import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()

    self.pos_encoding = self.positional_encoding(position, d_model)


  def get_angles(self, position, i, d_model):

    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles


  def positional_encoding(self, position, d_model):

    position=tf.range(position, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

    angle_rads = self.get_angles(position=position, i=i, d_model=d_model)

    angle_rads = np.zeros(angle_rads.shape)
    angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.constant(angle_rads)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    
    return tf.cast(pos_encoding, tf.float32)


  def call(self, inputs):

    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
