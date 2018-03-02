import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def model(y, t, k):
    dydt = -k * y
    return dydt


y0 = 5
t = np.linspace(0, 20)

y = odeint(model, y0, t, args=(1,))
plt.plot(t,y)
plt.show()