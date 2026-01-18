# from ipc import Channel
from struct import unpack, pack, calcsize
import curses
import re
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

def proj(a, b):
    return (np.dot(a, b)) / np.dot(a, a) * a

def simproj(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return np.dot(a, b).item() / np.linalg.norm(a)

class ShaderVarsProperties(type):

    # Camera Position
    @property
    def P(cls):
        if not cls.VSBOpen:
            return cls._P
        return np.array(cls.VSB(offset=(3*4*4), numbytes=4*3))

    # First model-view matrix vector (pointing Right)
    @property
    def R(cls):
        if not cls.VSBOpen:
            return cls._R
        return np.array(cls.VSB(offset=(7*4*4)+(0*4*4), numbytes=4*3))

    # Second model-view matrix vector (pointing Left)
    @property
    def L(cls):
        if not cls.VSBOpen:
            return cls._L
        return np.array(cls.VSB(offset=(7*4*4)+(1*4*4), numbytes=4*3))

    # Third model-view matrix vector (pointing Down)
    @property
    def D(cls):
        if not cls.VSBOpen:
            return cls._D
        return np.array(cls.VSB(offset=(7*4*4)+(2*4*4), numbytes=4*3))

    # Fourth model-view matrix vector (pointing Up)
    @property
    def U(cls):
        if not cls.VSBOpen:
            return cls._U
        return np.array(cls.VSB(offset=(7*4*4)+(3*4*4), numbytes=4*3))
    
    @property
    def SCREEN_X(cls):
        if not cls.VSBOpen:
            return cls._SCREEN_X
        return cls.VSB(offset=16*15, numbytes=4)

    @property
    def SCREEN_Y(cls):
        if not cls.VSBOpen:
            return cls._SCREEN_Y
        return cls.VSB(offset=16*15, numbytes=4)

    @property
    def X(cls):
        return (cls.R - cls.L) / 2

    @property
    def Y(cls):
        return (cls.U - cls.D) / 2
    
    @property
    def Z(cls):
        return (cls.R + cls.L) / 2

    @property
    def ROT(cls):
        return np.stack( (cls.X, cls.Y, cls.Z))

    @X.setter
    def X(cls, value):
        raise Exception(f"Attempted to set ShaderVars.X to {value}")

    @Y.setter
    def Y(cls, value):
        raise Exception(f"Attempted to set ShaderVars.Y to {value}")

    @Z.setter
    def Z(cls, value):
        raise Exception(f"Attempted to set ShaderVars.Z to {value}")

    @ROT.setter
    def ROT(cls, value):
        raise Exception(f"Attempted to set ShaderVars.ROT to {value}")


class ShaderVars(metaclass=ShaderVarsProperties):

    _R, _L, _D, _U = [np.array(v) for v in ([-0.674383, -0.723346, 0.14825], [0.834746, -0.530302, 0.14825], [0.0784705, -0.613447, -0.785827], [0.0259412, -0.202797, 0.978877])]
    _P = np.array((0, 0, 0), dtype=np.float32)
    _SCREEN_X = 1.0
    _SCREEN_Y = 1.0

    VSBOpen = False

    @staticmethod
    def VSB(num=2, offset=0, numbytes=None, T='f'):
        from ipc import Channel
        if not hasattr(ShaderVars, 'VSBFileLength'):
            ShaderVars.VSBFileLength = Channel(8, f"VSB2Length")
        sizeT = calcsize(T)
        filelen = unpack("@Q", ShaderVars.VSBFileLength.pop_nbl())[0]
        if filelen == 0:
            return None
        if not hasattr(ShaderVars, 'VSBChannel'):
            ShaderVars.VSBChannel = Channel(filelen, f"VSB2")
        if numbytes is None:
            numbytes = (filelen - offset) - ((filelen - offset) % sizeT)
        data = ShaderVars.VSBChannel.pop_nbl(offset=offset, numbytes=numbytes)
        n = numbytes // sizeT
        return unpack(f"@{n}{T}", data)

    @staticmethod
    def toggle():
        if not ShaderVars.VSBOpen:
            f = ShaderVars.VSB(numbytes=4)
            if f is not None:
                ShaderVars.VSBOpen = True
            else:
                return False
        else:
            ShaderVars.VSBOpen = False
    
class DummyShader:

    def __init__(self):
        self.ion = plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.color = 0
        self.elev, self.azim, self.roll = self.get_angle_offsets()
        self.ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
        self.ax.set_axis_off()
        self.zoom(1.0)
        self.quiver(ShaderVars.X / np.linalg.norm(ShaderVars.X), color='red', alpha=0.5)
        self.quiver(ShaderVars.Y / np.linalg.norm(ShaderVars.Y), color='blue', alpha=0.5)
        self.quiver(ShaderVars.Z / np.linalg.norm(ShaderVars.Z), color='green', alpha=0.5)
        self.quiver(ShaderVars.R / np.linalg.norm(ShaderVars.R), color='black', alpha=0.3)
        self.quiver(ShaderVars.L / np.linalg.norm(ShaderVars.L), color='black', alpha=0.3)
        self.quiver(ShaderVars.D / np.linalg.norm(ShaderVars.D), color='black', alpha=0.3)
        self.quiver(ShaderVars.U / np.linalg.norm(ShaderVars.U), color='black', alpha=0.3)

    def quiver(self, vec, origin=np.zeros(3), color=None, alpha=None):
        self.ax.quiver(*origin, *np.array(np.ravel(vec)), arrow_length_ratio=0, color=plt.cm.viridis.colors[self.color] if color is None else color, alpha=alpha)
        if color is None:
            self.color = (self.color + 61) % len(plt.cm.viridis.colors)

    def zoom(self, b):
        self.ax.set_xlim3d(-b, b)
        self.ax.set_ylim3d(-b, b)
        self.ax.set_zlim3d(-b, b)

    @staticmethod
    def get_angle_offsets():
        elev = np.arcsin((ShaderVars.R*simproj(ShaderVars.R, ShaderVars.X) + ShaderVars.L*simproj(-ShaderVars.L, ShaderVars.X))[2])
        azim = np.arccos(ShaderVars.X[1] / np.linalg.norm(ShaderVars.X))
        return [-np.degrees(elev), np.degrees(azim), 0]

def quiver(*args, **kwargs):
    if not hasattr(DummyShader, 'Shader'):
        DummyShader.Shader = DummyShader()
    DummyShader.Shader.quiver(*args, **kwargs)

def show(*args, **kwargs):
    if not hasattr(DummyShader, 'Shader'):
        DummyShader.Shader = DummyShader()
    plt.show(*args, **kwargs)


def PLANE2WORLD(x, y):
    X, Y=ShaderVars.X, ShaderVars.Y
    return (x * X / np.linalg.norm(X)) + (y * Y / np.linalg.norm(Y))
    return (x * X) + (y * Y)

def N_AT(x, y):
    R, L, D, U=ShaderVars.R, ShaderVars.L, ShaderVars.D, ShaderVars.U
    xp, yp = (1.0+x) / 2, (1.0+y) / 2.0
    n = (L + (xp * (R - L))) +  (D + (yp * (U - D)))
    return n / np.linalg.norm(n)

def TOWORLD(x, y, z, R=ShaderVars.R, L=ShaderVars.L, D=ShaderVars.D, U=ShaderVars.U, X=ShaderVars.X, Y=ShaderVars.Y):
    n = N_AT(x, y, R=R, L=L, D=D, U=U)
    offset = PLANE2WORLD(x, y, X=X, Y=Y)
    return np.ravel(n * z)

def FROMWORLD(x, y, z):
    pass

# TEST_PLANE_VECTOR = (-0.7, 0.5, 1.5)
# P2W = PLANE2WORLD(*TEST_PLANE_VECTOR[:2])
# PLANENORM = N_AT(*TEST_PLANE_VECTOR[:2])
# PLANEX = ShaderVars.X / np.linalg.norm(ShaderVars.X)
# PLANEX = PLANEX - proj(PLANEX, PLANENORM) - proj(PLANENORM, PLANEX)
# print(P2W)
# quiver(PLANENORM, origin=P2W)
# quiver(PLANEX, origin=P2W)
# print( np.dot(ShaderVars.Y, P2W) / np.linalg.norm(ShaderVars.Y) )
# show(block=True)
# exit()


from ipc import Channel
N_ARROWS = 3
ARROW_COLORS = {
    'red': (255, 0, 0, 255), 
    'green': (0, 255, 0, 255), 
    'blue': (0, 0, 255, 255), 
    'yellow': (255, 255, 0, 255), 
    'teal': (0, 255, 255, 255),
    'purple': (255, 0, 255, 255)
}
_ARROW_COLORS = ['red', 'green', 'blue', 'yellow', 'teal', 'purple']
COLOR_INDEX = 0
ARROW_INDEX = 0
# NumArrows = Channel(calcsize('@I'), "NUM_DEBUG_ARROWS")
# NumArrows.push_nbl(pack('@I', N_ARROWS))

DebugArrows = [Channel(6*calcsize('@f'), f"DebugArrow{i}") for i in range(N_ARROWS)]
ArrowColors = [Channel(4*calcsize('@i'), f"DebugArrowColors{i}") for i in range(N_ARROWS)]
ShaderVars.toggle()

def push_arrow(start, end, index, color='blue'):
    global COLOR_INDEX, ARROW_INDEX
    if color is None:
        color = _ARROW_COLORS[COLOR_INDEX]
        COLOR_INDEX = (COLOR_INDEX + 1) % len(_ARROW_COLORS)
    if index == -1:
        index = ARROW_INDEX
        ARROW_INDEX = (COLOR_INDEX + 1) % N_ARROWS
    color = ARROW_COLORS[color]
    DebugArrows[index].push_nbl(pack('@6f', *start, *end))
    ArrowColors[index].push_nbl(pack('@4i', *color))

print("Trying channel")
while not hasattr(ShaderVars, 'VSBChannel'):
    print(ShaderVars.R)
    time.sleep(1.0)

pos1 = np.array((0.1, 0.1))
pos2 = -pos1
print("running")
while True:
    X, Y, Z = ShaderVars.X, ShaderVars.Y, ShaderVars.Z
    R, L, D, U = ShaderVars.R, ShaderVars.L, ShaderVars.D, ShaderVars.U
    plane_pos1 = PLANE2WORLD(*pos1)
    # plane_pos2 = PLANE2WORLD(*pos2)
    zvec = N_AT(*pos1)
    # z1 = (plane_pos1 + zvec, plane_pos1 + 2*zvec)
    # z2 = (plane_pos2 + Z, plane_pos2 + Z + Z)
    # print(ShaderVars.X)
    # print(ShaderVars.ROT)

    # print(ShaderVars.P + plane_pos1)

    push_arrow(ShaderVars.P + 0.1 * Z + plane_pos1, Z, index=0, color='blue')
    push_arrow(ShaderVars.P + 0.1 * Z, Y, index=1, color='green')
    push_arrow(ShaderVars.P + 0.1 * Z, Z, index=2, color='red')
    # push_arrow(*z2, color='blue')
    # push_arrow(Z, Z + Z)
    # push_arrow(Z, Z + Y)
    # push_arrow(Z, Z + X)
N = 3

VSBLengths = [Channel(8, f"VSB{i}Length") for i in range(N)]
VSBLen = lambda i: unpack("@Q", VSBLengths[i].pop_nbl())[0]
VSBFiles = [Channel(VSBLen(i), f"VSB{i}") for i in range(N)]
VSBView = lambda i, length=None, offset=0: VSBFiles[i].pop_nbl()[offset:(None if length is None else offset+length)]

def VSBViewType(i, n=None, offset=0, T='f'):
    if n is None:
        n = (VSBLen(i) - offset) // 4
    return unpack(f"@{n}{T}", VSBView(i, length=n*4, offset=offset))

def try_inverse():

    # D = VideoState.pop_depth()[1, ...].squeeze()

    VBS = [np.array(VSBViewType(i)) for i in range(3)]
    MVP = np.asmatrix(VBS[2][28:28+16].reshape(4, 4))

    ROT = MVP
    ROT[:, 3] = 0

    SCREEN_X = VBS[2][4*15]
    SCREEN_Y = VBS[2][(4*15)+1]

# line 15: window size
def main(stdscr):

    stdscr.nodelay(True)

    # R, G, B = Channel(12, "RVector"), Channel(12, "GVector"), Channel(12, "BVector")

    def writevec(vec: Channel, arr):
        vec.push_nbl(pack('@3f', *np.ravel(arr)))

    VAL_LEN = 10
    COL_LEN = 14

    n = 0

    while True:
        VBS = [np.array(VSBViewType(i)) for i in range(3)]
        MVP = np.asmatrix(VBS[2][28:28+16].reshape(4, 4))

        Lv, Rv, Dv, Uv = [np.ravel(x) for x in [MVP[0, :3], MVP[1, :3], MVP[2, :3], MVP[3, :3]]]
        
        X = np.cross(Dv, Uv)
        Y = np.cross(Lv, Rv)
        Z = -np.cross(X, Y)

        ROT = np.stack((ShaderVars.R, ShaderVars.L, ShaderVars.D, ShaderVars.U))

        push_arrow(ShaderVars.P + Z, ShaderVars.R, color='red', index=0)
        push_arrow(ShaderVars.P + Z, ShaderVars.L, color='blue', index=1)
        # push_arrow(Z, ShaderVars.Z, color='green', index=2)


        stdscr.clear()
        
        I = 2
        for r in range(len(VBS[I]) // 4):
            for c in range(4):
                stdscr.addstr(r, COL_LEN*c, str(VBS[I][4*r+c])[0:VAL_LEN])

        for c in range(3):
            stdscr.addstr(r+2, COL_LEN*c, str(ROT[0, c])[0:VAL_LEN])
            stdscr.addstr(r+3, COL_LEN*c, str(ROT[1, c])[0:VAL_LEN])
            stdscr.addstr(r+4, COL_LEN*c, str(ROT[2, c])[0:VAL_LEN])
            stdscr.addstr(r+5, COL_LEN*c, str(ROT[3, c])[0:VAL_LEN])

        # stdscr.addstr(r+1, 0, f'{SCREEN_X} {SCREEN_Y}')
        stdscr.refresh()

curses.wrapper(main)
