import matplotlib.pyplot as plt
import numpy as np

class DummyShader:

    def __init__(self):
        self.ion = plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.color = 0
        # self.elev, self.azim, self.roll = self.get_angle_offsets()
        # self.ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
        self.ax.set_axis_off()
        self.zoom(1.0)

    def quiver(self, vec, origin=np.zeros(3), color=None, alpha=None):
        self.ax.quiver(*origin, *np.array(np.ravel(vec)), arrow_length_ratio=0, color=plt.cm.viridis.colors[self.color] if color is None else color, alpha=alpha)
        if color is None:
            self.color = (self.color + 61) % len(plt.cm.viridis.colors)

    def zoom(self, b):
        self.ax.set_xlim3d(-b, b)
        self.ax.set_ylim3d(-b, b)
        self.ax.set_zlim3d(-b, b)

def quiver(*args, **kwargs):
    if not hasattr(DummyShader, 'Shader'):
        DummyShader.Shader = DummyShader()
    DummyShader.Shader.quiver(*args, **kwargs)

def show(*args, **kwargs):
    if not hasattr(DummyShader, 'Shader'):
        DummyShader.Shader = DummyShader()
    plt.show(*args, **kwargs)

