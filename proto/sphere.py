import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


vec = sample_spherical(1000).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xi,yi,zi, s=0.2)
plt.show()