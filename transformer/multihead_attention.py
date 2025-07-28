import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # d_model=512, num_heads=8, depth=64
        self.depth = d_model // self.num_heads
    
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)


    def split_heads(self, inputs, batch_size):

        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(inputs, perm=[0, 2, 1, 3])
  
  
    def call(self, inputs):

      query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']

      batch_size = tf.shape(query)[0]

    # q: (batch_size, query sequence, d_model)
    # k: (batch_size, key sequence, d_model)
    # v: (batch_size, value sequence, d_model)
    query = self.query_dense(query)
    key = self.key_dense(key) 
    value = self.value_dense(value)
          
    # q: (batch_size, num_heads, query sequence, d_model/num_heads)
    # k: (batch_size, num_heads, key sequence, d_model/num_heads)
    # v: (batch_size, num_heads, value sequence, d_model/num_heads)
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    #(batch_size, num_heads, query sequence, d_model/num_heads)
    scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)

    #(batch_size, query sequence, num_heads, d_model/num_heads)  
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    
    #(batch_size, query sequence, d_model)
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

    #(batch_size, query의 sequence, d_model)
    return self.dense(concat_attention)    


def scaled_dot_product_attention(query, key, value, mask):

  #q: (batch_size, num_heads, query sequence, d_model/num_heads)
  #k: (batch_size, num_heads, key sequence, d_model/num_heads)
  #v: (batch_size, num_heads, value sequence, d_model/num_heads)
  #padding_mask : (batch_size, 1, 1, key sequence)

  matmul_qk = tf.matmul(query, key, transpose_b=True)

  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)
  
  if mask is not None:
    logits += (mask * -1e9)
  
  #(batch_size, num_heads, query sequence, key sequence)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  #(batch_size, num_heads, query sequence, d_model/num_heads)
  output = tf.matmul(attention_weights, value)

  return output, attention_weights
