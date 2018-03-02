from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


def getSamples(n):
    # generate vector of random angles
    angles = np.random.uniform(-np.pi, np.pi, n)

    # generate matrix of x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    return angles, x, y


angles, x_pos, y_pos = getSamples(10000)
x_train = (np.array([x_pos[-9000:], y_pos[-9000:]]).T + 1) / 2
x_test = (np.array([x_pos[:1000], y_pos[:1000]]).T + 1) / 2

# Dimension of z space
z_dim = 1
dense_dim = 1024
data_dim = 2

inputs = Input(shape=(data_dim, ))
dense_in = Dense(dense_dim, activation='relu')(inputs)
latent = Dense(z_dim, activation='relu', name='latent_space')(dense_in)
dense_out = Dense(dense_dim, activation='relu')(latent)
outputs = Dense(data_dim, activation='sigmoid', name='output')(dense_out)

# Build autoencoder directly from layers
autoencoder = Model(inputs, outputs)

# separate encoder for testing afterwards
encoder = Model(inputs, latent)

# We build decoder from existing graph
latent_input = Input(shape=(z_dim, ))

# Grab shared layers from the autoencoder and nest them starting from the dense_out layer
decoder = Model(latent_input,
                autoencoder.layers[-1](autoencoder.layers[-2](latent_input)))

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test))
# Check
# encode and decode some digits
# note that we take them from the *test* set
encoded_val = encoder.predict(x_test*2 -1)
decoded_val = decoder.predict(encoded_val)

plt.scatter(x_test[:, 0], x_test[:, 1])
plt.scatter(decoded_val[:, 0], decoded_val[:, 1])
plt.show()
