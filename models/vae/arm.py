# Simple implementation of a n-dof arm
# We will first start with 1 dof
import numpy as np


def getSamples(n):
    # generate vector of random angles
    angles = np.random.uniform(-np.pi,np.pi, n)

    # generate matrix of x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    return angles, x, y
