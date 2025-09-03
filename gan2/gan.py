import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#tensorflow ver: 2.12.0
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.initializers import RandomNormal

noise_dim=100
epochs=30
batch_size=256

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_x = train_x / 127.5 - 1
train_x = train_x.reshape(-1,28 * 28)
test_x = test_x / 127.5 - 1

dataset=tfds.load(name='fashion_mnist', split=tfds.Split.TRAIN)
dataset=dataset.map(lambda x: tf.cast(x['image'], tf.float32) / 127.5 - 1).batch(batch_size)

generator = tf.keras.models.Sequential([
    Dense(256, input_dim=noise_dim),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(28*28, activation='tanh'),
])


discriminator = tf.keras.models.Sequential([
    Dense(1024, input_shape=(784,), kernel_initializer=RandomNormal(stddev=0.02)),
    LeakyReLU(0.2), 
    Dropout(0.3), 
    Dense(512),
    LeakyReLU(0.2), 
    Dropout(0.3), 
    Dense(256),
    LeakyReLU(0.2), 
    Dropout(0.3), 
    Dense(1, activation='sigmoid')
])

optm_g = tf.keras.optimizers.Adam(0.0002, 0.5)
optm_d = tf.keras.optimizers.Adam(0.0002, 0.5)

discriminator.compile(loss='binary_crossentropy', optimizer=optm_d)

discriminator.trainable = False
gan_input = Input(shape=(noise_dim,))
x = generator(inputs=gan_input)
output = discriminator(x)

gan=tf.keras.Model(gan_input,output)
gan.compile(loss='binary_crossentropy',optimizer=optm_g)
#gan.summary()

def get_batches(data, batch_size):
    batches = []
    for i in range(int(data.shape[0] // batch_size)):
        batch = data[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)
    return np.asarray(batches)


def save_image(image, path):
    plt.figure(figsize=(8, 4))
    for i in range(fake.shape[0]):
        plt.subplot(4, 6, i+1)
        plt.imshow(image[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


for epoch in range(0, epochs):

    d_losses = []
    g_losses = []
    
    for real in get_batches(train_x, batch_size):

        noise = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
        
        fake = generator.predict(noise, verbose=0)
        
        x_dis = np.concatenate([real, fake])

        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 1.0

        discriminator.trainable = True

        d_loss = discriminator.train_on_batch(x_dis, y_dis)
        
        noise = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
        y_gan = np.ones(batch_size)

        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gan)

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
    if (epoch + 1) % 5 == 0:

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % ((epoch + 1), epochs, np.asarray(d_losses).mean(), np.asarray(g_losses).mean()))        

        noise = np.random.normal(0, 1, size=(24, noise_dim))
        fake = generator.predict(noise, verbose=0).reshape(-1, 28, 28)
        
        fake_path = "fake"
        os.makedirs(fake_path, exist_ok=True)
        
        save_image(fake, (fake_path + "/%d.png" % (epoch + 1)))
