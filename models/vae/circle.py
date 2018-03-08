import arm
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
angles, x,y = arm.getSamples(1000)

plt.scatter(x,y)
plt.show()
