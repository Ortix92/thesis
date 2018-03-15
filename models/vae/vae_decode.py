import matplotlib.pyplot as plt
import keras

# Encode datapoints
means = encoder.predict(x_test)
plt.figure()
plt.title('Encoded means')
plt.ylabel('z2')
plt.xlabel('z1')
plt.scatter(means[:, 0], means[:, 1], s=0.5, alpha=0.5)

# Decode means
[decoded_mean, decoded_variance] = decoder.predict(means)
plt.figure()
plt.title('Reconstructed means')
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(decoded_mean[:, 0], decoded_mean[:, 1], s=0.5, c="red")

# Generate random points
plt.figure()
plt.title('Random point generation from random latent space')
plt.ylabel('x2')
plt.xlabel('x1')

random_means = np.random.normal(loc=0, scale=1.0, size=[10000, 2])
[decoded_random_means,
 decoded_random_variances] = decoder.predict(random_means)
decoded_random = decoded_random_means + np.exp(
    decoded_random_variances / 2) * np.random.normal(
        loc=0, scale=1.0, size=[10000, 2])
plt.scatter(decoded_random[:, 0], decoded_random[:, 1], s=0.5, c="orange")
plt.show()