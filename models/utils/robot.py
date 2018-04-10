import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Robot:
    def __init__(self, angles=[0]):
        self.angles = angles

        # Init plot
        fig = plt.figure()
        ax = fig.add_subplot(
            111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.line, = ax.plot([], [], 'o-', lw=2)

    def setAngles(self, angles):
        if(isinstance(angles,list)):
            self.angles = angles
        else:
            self.angles = [angles]

    def eef(self):
        x = np.cos(self.angles[0])
        y = np.sin(self.angles[0])
        return x, y

    def draw(self):
        x, y = self.eef()
        thisx = [0, x]
        thisy = [0, y]
        self.line.set_data(thisx, thisy)
        plt.show()
