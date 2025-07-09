import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from loader import *
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from bahdanau_attention import BahdanauAttention
import time


def tokenize(lang):

  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples, head, tail):

  targ_lang, inp_lang = create_dataset(path, num_examples, head, tail)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


file_path = "english_spain.txt"
num_examples = 30000
head = '<start>'
tail = '<end>'

input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(file_path, num_examples, head, tail)

max_length_inp = input_tensor.shape[1]
max_length_targ = target_tensor.shape[1]

batch_size = 64
steps_per_epoch = len(input_tensor) // batch_size
embedding_dim = 256
units = 1024

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
dataset = dataset.batch(batch_size, drop_remainder=True)

vocab_inp_size = len(inp_lang.word_index) + 1
encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)

attention_layer = BahdanauAttention(units)

vocab_tar_size = len(targ_lang.word_index) + 1
decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size, attention_layer)

optimizer = tf.keras.optimizers.Adam()

def model_tensor_shape():

  example_input_batch, example_target_batch = next(iter(dataset))
  example_input_batch.shape, example_target_batch.shape

  sample_hidden = encoder.initialize_hidden_state()
  sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
  print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
  print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

  attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

  print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
  print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

  sample_decoder_output, _, _ = decoder(tf.random.uniform((batch_size, 1)), sample_hidden, sample_output)
  print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


def loss_function(loss_object, real, pred):

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):

  loss = 0

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

  with tf.GradientTape() as tape:

    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index[head]] * batch_size, 1)
    
    for t in range(1, targ.shape[1]):
      
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(loss_object, targ[:, t], predictions)
      
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables  

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


def train():

  epoch_count = 3

  for epoch in range(epoch_count):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

      batch_loss = train_step(inp, targ, enc_hidden)
      total_loss += batch_loss

      if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
  
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def test(sentence):
  
  sentence = preprocess_sentence(sentence, head, tail)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
  inputs = tf.convert_to_tensor(inputs)

  predict = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index[head]], 0)

  for t in range(max_length_targ):

    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

    attention_weights = tf.reshape(attention_weights, (-1, ))

    predicted_id = tf.argmax(predictions[0]).numpy()

    predict += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == tail:
      return predict, sentence

    dec_input = tf.expand_dims([predicted_id], 0)

  return predict, sentence


def translate(sentence):

  predict, sentence = test(sentence)

  print('Input: %s' % (sentence))
  print('Predict: {}'.format(predict))


model_tensor_shape()
train()

translate(u'hace mucho frio aqui.')
translate(u'esta es mi vida.')
translate(u'¿todavia estan en casa?')
translate(u'trata de averiguarlo.')