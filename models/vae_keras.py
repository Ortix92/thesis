from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.utils import plot_model
plt.ioff()
batch_size = 10
original_dim = 2
latent_dim = 2
intermediate_dim = 1024
epochs = 50
epsilon_std = 1.0

x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def getSamples(n):
    # generate vector of random angles
    angles = np.random.uniform(-np.pi, np.pi, n)

    # generate matrix of x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    return angles, x, y


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
kl_loss = -0.5 * K.sum(
    1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# Retrieve testing data
angles, x_pos, y_pos = getSamples(1000)
x_train = (np.array([x_pos[-900:], y_pos[-900:]]).T+1)/2
x_test = (np.array([x_pos[:100], y_pos[:100]]).T+1)/2
vae.fit(
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None))

# build a x-y generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim, ))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

x_gen = np.zeros((1000, 2))
for i in range(1000):
    x_gen[i] = generator.predict(
        np.array([[np.random.normal(0, 1),
                   np.random.normal(0, 1)]]))

plt.scatter(x_gen[:, 0], x_gen[:, 1])
plt.scatter(x_pos,y_pos)
plt.show()
