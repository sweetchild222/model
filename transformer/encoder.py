import tensorflow as tf
from positional_encoding import PositionalEncoding
from multihead_attention import MultiHeadAttention


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
  
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")  
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
            'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    # Position-wise FFNN
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Drop out + residual learning
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    
    inputs = tf.keras.Input(shape=(None,), name="inputs")
  
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # Stack encorder
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
