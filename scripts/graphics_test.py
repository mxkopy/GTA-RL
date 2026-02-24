from ipc import Channel, StructuredChannel
from struct import unpack, pack, calcsize
import curses
import numpy as np
import matplotlib.pyplot as plt

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
        return cls.VSB(offset=16*15, numbytes=4)[0]

    @property
    def SCREEN_Y(cls):
        if not cls.VSBOpen:
            return cls._SCREEN_Y
        return cls.VSB(offset=16*15 + 4, numbytes=4)[0]

    @property
    def NEAR_CLIP(cls):
        if not cls.VSBOpen:
            return cls._NEAR_CLIP
        return unpack('f', cls.NearFar.pop_nbl(numbytes=calcsize('f')))[0]

    @property
    def FAR_CLIP(cls):
        if not cls.VSBOpen:
            return cls._FAR_CLIP
        return unpack('f', cls.NearFar.pop_nbl(offset=calcsize('f'), numbytes=calcsize('f')))[0]
    
    # There's some 3D graphics rendering reason for why this is needed
    # that I can't figure out 
    @property
    def VIEWPORT_DIMENSIONS(cls):
        if not cls.VSBOpen:
            return 1, 1
        return cls.VSB(offset=16*5, numbytes=8)

    @property
    def X(cls):
        X = cls.R - cls.L
        return X / np.linalg.norm(X)

    @property
    def Y(cls):
        Y = cls.U - cls.D
        return Y / np.linalg.norm(Y)
    
    @property
    def Z(cls):
        Z = cls.R + cls.L
        return Z / np.linalg.norm(Z)
    
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
    _NEAR_CLIP = 0.0
    _FAR_CLIP = 1.0

    VSBOpen = False
    NearFar = Channel(calcsize('2f'), "NearClipFarClip")

    @staticmethod
    def VSB(num=2, offset=0, numbytes=None, T='f'):
        if not hasattr(ShaderVars, f'VSB{num}FileLength'):
            setattr(ShaderVars, f'VSB{num}FileLength', Channel(8, f"VSB{num}Length"))        
        sizeT = calcsize(T)
        filelen = unpack("@Q", getattr(ShaderVars, f'VSB{num}FileLength').pop_nbl())[0]
        if filelen == 0:
            return None
        if not hasattr(ShaderVars, f'VSB{num}'):
            setattr(ShaderVars, f'VSB{num}', Channel(filelen, f"VSB{num}"))
        if numbytes is None:
            numbytes = (filelen - offset) - ((filelen - offset) % sizeT)
        data = getattr(ShaderVars, f'VSB{num}').pop_nbl(offset=offset, numbytes=numbytes)
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

    def __init__(self, default_arrows=False):
        self.ion = plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.color = 0
        self.elev, self.azim, self.roll = self.get_angle_offsets()
        self.ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
        self.ax.set_axis_off()
        self.zoom(1.0)
        if default_arrows:
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

def RAY_AT_NDC(x, y, z=1.0):
    vx, vy = ShaderVars.VIEWPORT_DIMENSIONS
    sx, sy = ShaderVars.SCREEN_X, ShaderVars.SCREEN_Y
    x = x * (sx/vx)
    y = y * (sy/vy)
    return ShaderVars.ROT.T @ np.array((x, y, z))
    

def PLANE2WORLD(x, y):
    return RAY_AT_NDC(x, y, z=0.0)

def FROMWORLD(x, y, z):
    pass

# print(2.2689274224044755 / 1.7288218284951495)
# print(1276 / 697)
# exit()
# print(np.linalg.norm(ShaderVars.L), np.linalg.norm(ShaderVars.R))
# print(np.acos(np.dot(ShaderVars.U, ShaderVars.D)) * 180 / np.pi)
# exit()
# TEST_PLANE_VECTOR = (0.5, 0.5, 1.5)
# P2W = PLANE2WORLD(*TEST_PLANE_VECTOR[:2])
# NORM = N_AT(*TEST_PLANE_VECTOR[:2])
# quiver(NORM, origin=P2W)
# show(block=True)
# exit()

from matplotlib.colors import to_rgba, BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS
class DebugArrows:
    
    N_ARROWS = 3

    INDEX = 0

    INDEX_TO_COLOR = {
        index: tuple(round(c*255) for c in to_rgba(color)) for index, color in zip(range(N_ARROWS), BASE_COLORS)
    }

    ARROWS = {
        i: StructuredChannel(float, float, float, float, float, float, tagname=f"DebugArrow{i}") for i in range(N_ARROWS)
    }

    COLORS = {
        i: StructuredChannel(int, int, int, int, tagname=f"DebugArrowColors{i}") for i in range(N_ARROWS)
    }

    @staticmethod
    def push_arrow(start, end, index=None):
        if index is None:
            index = DebugArrows.INDEX
            DebugArrows.INDEX = (DebugArrows.INDEX + 1) % DebugArrows.N_ARROWS
        DebugArrows.ARROWS[index].push_nbl(*start, *end)
        # DebugArrows.COLORS[index].push_nbl(*DebugArrows.INDEX_TO_COLOR[index])
        DebugArrows.COLORS[index].push_nbl(255, 0, 0, 255)


from math import floor
import time

class RayCastGetItem(type):

    def __getitem__(cls, idx):
        r, c = idx
        if (r, c) not in cls.RAYS:
            cls.RAYS[(r, c)] = StructuredChannel(float, float, float, tagname=f"Ray{c}_{r}")
        collision = np.array(cls.RAYS[(r, c)].pop_nbl())
        depth = cls.DEPTH[r, c, 3]
        return collision, depth

ShaderVars.toggle()
from environment import VideoState
if ShaderVars.VSBOpen:
    VideoState.init_cuda_array("DepthBuffer")

class RayCasts(metaclass=RayCastGetItem):
    RAYS = {}
    DEPTH = None if "DepthBuffer" not in VideoState.cuda_arrays else VideoState.cuda_arrays["DepthBuffer"]
    UPDATE = StructuredChannel(bool, tagname="RayCastUpdate")
    DEFAULT_PIXELCOORDS = None if not ShaderVars.VSBOpen else [(int(floor(r)), int(floor(c))) for (r, c) in [ 
        (ShaderVars.SCREEN_Y // 4, ShaderVars.SCREEN_X // 4),
        (ShaderVars.SCREEN_Y // 4, (ShaderVars.SCREEN_X // 4) + (ShaderVars.SCREEN_X // 2)),
        ((ShaderVars.SCREEN_Y // 2) + (ShaderVars.SCREEN_Y // 4), ShaderVars.SCREEN_X // 4),
        ((ShaderVars.SCREEN_Y // 2) + (ShaderVars.SCREEN_Y // 4), (ShaderVars.SCREEN_X // 2) + (ShaderVars.SCREEN_X // 4))
    ]]



def try_raycast():
    RayCasts.UPDATE.push_nbl(True)
    while RayCasts.UPDATE.pop_nbl():
        time.sleep(0.1)
    raycastinfo = {
        'P': ShaderVars.P,
        'NearClip': ShaderVars.NEAR_CLIP,
        'FarClip': ShaderVars.FAR_CLIP,
        'VSB1': ShaderVars.VSB(num=1),
        'VSB2': ShaderVars.VSB(num=2),
        'Rays': []
    }
    for r, c in RayCasts.DEFAULT_PIXELCOORDS:
        collision, depth = RayCasts[r, c]
        ray = {
            'pixel': (c, r),
            'collision': tuple(float(x) for x in tuple(collision)),
            'depth': depth.get()
        }
        raycastinfo['Rays'].append(ray)
    return raycastinfo

def save_raycast():
    import pickle
    raycastinfo = try_raycast()
    with open('raycastinfo.pickle', 'wb') as file:
        pickle.dump(raycastinfo, file)

def load_raycast():
    import pickle
    with open('raycastinfo.pickle', 'rb') as file:
        return pickle.load(file)

# line 15: window size
def main(stdscr):

    stdscr.nodelay(True)

    VAL_LEN = 10
    COL_LEN = 14

    n = 0
    pos = (0, 0)
    d = 100

    while True:
        try:
            key = stdscr.getkey()
            if key == 'KEY_LEFT':
                pos = pos[0]-1, pos[1]
            if key == 'KEY_RIGHT':
                pos = pos[0]+1, pos[1]
            if key == 'KEY_DOWN':
                pos = pos[0], pos[1]-1
            if key == 'KEY_UP':
                pos = pos[0], pos[1]+1
            if key == 'S':
                d = d - 1
            if key == 'W':
                d = d + 1
        except:
            pass

        p = pos[0] / 100, pos[1] / 100

        RAY = RAY_AT_NDC( 
            2.0*(319.0 / ShaderVars.SCREEN_X) - (ShaderVars.VIEWPORT_DIMENSIONS[0] / ShaderVars.SCREEN_X), 
            2.0*(522.0 / ShaderVars.SCREEN_X) - (ShaderVars.VIEWPORT_DIMENSIONS[0] / ShaderVars.SCREEN_X)
        )

        DebugArrows.push_arrow(ShaderVars.P + ShaderVars.Z, ShaderVars.Z, index=0)

        # V = RAY + (0.3 * RAY * (ShaderVars.FAR_CLIP - ShaderVars.NEAR_CLIP) / ShaderVars.NEAR_CLIP)

        # DebugArrows.push_arrow(ShaderVars.P + ShaderVars.Z, ShaderVars.Z, index=0)
        # DebugArrows.push_arrow(ShaderVars.P + RAY, RAY, index=1)

        VertexShaderBuffer = ShaderVars.VSB()
        stdscr.clear()
        
        for r in range(len(VertexShaderBuffer) // 4):
            for c in range(4):
                stdscr.addstr(r, COL_LEN*c, str(VertexShaderBuffer[4*r+c])[0:VAL_LEN])
        stdscr.addstr(r+2, 0, str(p + (d / 100, )))
        stdscr.refresh()

if __name__ == '__main__':
    curses.wrapper(main)


