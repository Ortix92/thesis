from keras.layers import Input, Dense
from keras.callbacks import TensorBoard
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


n = 10000
angles, x_pos, y_pos = getSamples(n)
pos_arr = (np.array([x_pos, y_pos]).T)
x_train = np.c_[np.array(angles[int(-n * 0.9):]), pos_arr[int(-n * 0.9):]]
x_test = np.c_[np.array(angles[int(n * 0.1):]), pos_arr[int(n * 0.1):]]

x_train = (pos_arr[int(-n * 0.9):] + 1) / 2
x_test = (pos_arr[int(n * 0.1):] + 1) / 2

# Dimension of z space
z_dim = 1
dense_dim = 256
data_dim = 2

inputs = Input(shape=(data_dim, ))
encoded = Dense(dense_dim, activation='relu')(inputs)
encoded = Dense(dense_dim, activation='relu')(encoded)
encoded = Dense(z_dim, activation='relu', name='latent_space')(encoded)
decoded = Dense(dense_dim, activation='relu')(encoded)
decoded = Dense(dense_dim, activation='relu')(decoded)
outputs = Dense(data_dim, activation='sigmoid', name='output')(decoded)

# Build autoencoder directly from layers
autoencoder = Model(inputs, outputs)

# separate encoder for testing afterwards
encoder = Model(inputs, encoded)

# We build decoder from existing graph
latent_input = Input(shape=(z_dim, ))

# Grab shared layers from the autoencoder and nest them starting from the dense_out layer
decoder = Model(latent_input,
                autoencoder.layers[-1](autoencoder.layers[-2](autoencoder.layers[-3](latent_input))))

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(
    x_train,
    x_train,
    epochs=150,
    batch_size=150,
    shuffle=True,
    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
    validation_data=(x_test, x_test))
# Check
# encode and decode some digits
# note that we take them from the *test* set
encoded_val = encoder.predict(x_test)
decoded_val = decoder.predict(encoded_val)

plt.scatter(x_test[:, 0], x_test[:, 1])
plt.scatter(decoded_val[:, 0], decoded_val[:, 1])
plt.show()
