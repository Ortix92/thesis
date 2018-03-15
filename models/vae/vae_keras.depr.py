'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 2
latent_dim = 2
intermediate_dim = 128
epochs = 50
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
optimizer = keras.optimizers.rmsprop(lr=0.0002)
vae.compile(optimizer=optimizer)
vae.summary()
def getSamples(n):
    # generate vector of random angles
    angles = np.random.uniform(-np.pi, np.pi, n)

    # generate matrix of x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    return angles, x, y

n = 10000
angles, x_pos, y_pos = getSamples(n)
# Scale and translate position to conform with sigmoid layer
pos_arr = (np.array([x_pos, y_pos]).T + 1) / 2
x_train = (pos_arr[int(-n * 0.9):])
x_test = (pos_arr[int(n * 0.1):])

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

random_means = np.random.normal(loc=0, scale=1.0, size=[10000, 2])

[decoded_random_means,
 decoded_random_variances] = generator.predict(random_means)
decoded_random = decoded_random_means + np.exp(
    decoded_random_variances / 2) * np.random.normal(
        loc=0, scale=1.0, size=[10000, 2])
plt.scatter(decoded_random[:, 0], decoded_random[:, 1], s=0.5, c="orange")
plt.show()