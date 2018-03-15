from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
from keras.layers import Input, Dense, Lambda, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers, losses
from keras.callbacks import TensorBoard
from keras.datasets import mnist

batch_size = 250
original_dim = 2
latent_dim = 2
intermediate_dim = 256 * 4
epochs = 500
epsilon_std = 1.0
learning_rate = 0.00018

# Sample unit circle
def getSamples(n):
    # generate vector of random angles
    angles = np.random.uniform(-np.pi, np.pi, n)

    # generate matrix of x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    return angles, x, y


# Implementation of the reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Encoder section
x = Input(shape=(original_dim, ))
x_norm = BatchNormalization(axis=1)(x)
h = Dense(intermediate_dim, activation='relu')(x_norm)
h_norm = BatchNormalization(axis=1)(h)
z_mean = Dense(latent_dim)(h_norm)
z_log_var = Dense(latent_dim)(h_norm)

# A lambda function/layer for the latent space from which we sample during decoding
z = Lambda(sampling)([z_mean, z_log_var])

# we instantiate these layers separately so we can reuse them later
z_norm = BatchNormalization(axis=1)(z)
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid', name='output')
h_decoded = decoder_h(z_norm)
h_decoded_norm = BatchNormalization(axis=1)(h_decoded)
x_decoded_mean = decoder_mean(h_decoded_norm)

# instantiate VAE model
vae = Model(x, x_decoded_mean)


# Compute total loss
def vae_loss(x, x_decoded_mean):
    xent_loss = losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return (xent_loss + kl_loss)

optimizer = optimizers.adam(lr=learning_rate)
vae.compile(optimizer,loss=vae_loss)
vae.summary()

# Create training and test data from circle samples
n = 10000
angles, x_pos, y_pos = getSamples(n)

# Scale and translate position to conform with sigmoid layer
pos_arr = (np.array([x_pos, y_pos]).T + 1) / 2
x_train = (pos_arr[int(-n * 0.9):])
x_test = (pos_arr[int(n * 0.1):])

vae.fit(
    x_train,
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[history],)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim, ))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# Sample from noise and generate new points
if (latent_dim == 1):
    noise = np.random.normal(size=(n, 1))
else:
    noise = np.random.multivariate_normal(
        np.zeros((latent_dim)), np.eye(latent_dim), n)

x_decoded = np.zeros((n, original_dim))
for i in range(n):
    z_sample = np.array([noise[i, :]])
    x_decoded[i, :] = generator.predict(z_sample)

print('Displaying graph')
plt.scatter(x_decoded[:, 0], x_decoded[:, 1])
plt.show()
