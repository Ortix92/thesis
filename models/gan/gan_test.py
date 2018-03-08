import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

generator = load_model('gan_trained.h5')
generator.summary()

# Generate samples from noise
noise = np.random.normal(0,1,(10000,100))
samples = generator.predict(noise)
samples = np.squeeze(samples,axis=2)
print(samples.shape)

plt.scatter(samples[:,0],samples[:,0])
plt.show()
