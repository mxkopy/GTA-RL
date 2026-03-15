from structs import *
from ipc import StructuredMemory, Flags
from environment import VideoState
import matplotlib.pyplot as plt
import sys
import pickle
from dummy_shader import *

FLAGS = Flags()
V = StructuredMemory("VSConstants")
RAYS = [StructuredMemory(f'Ray{X}') for X in ('A', 'B', 'C', 'D')]

def Proj():
    return np.frombuffer(V.data.constant_buffers[1], dtype=np.float32)

def VSB():
    return np.frombuffer(V.data.constant_buffers[2], dtype=np.float32)

def Axes():
    vsb = VSB()
    X = vsb[28:31] - vsb[32:35]
    Y = vsb[40:43] - vsb[36:39]
    Z = vsb[16:19]
    X = X / np.linalg.norm(X)
    Y = Y / np.linalg.norm(Y)
    return np.stack((X, Y, Z))

def Rays():
    FLAGS.wait_until(FLAGS.RAYCASTS, True)
    vsb = VSB()
    axes = Axes()
    depth = VideoState.pop()
    def get_depth(x, y):
        ar = depth.shape[2] / depth.shape[1]
        x = x / ar
        c = int(depth.shape[2] * ((x / 2) + 0.5))
        r = int(depth.shape[1] * ((y / 2) + 0.5))
        return float(depth[0, r, c])
    rays = [
        {
            'Ray': (ray.data.collision.x, ray.data.collision.y, ray.data.collision.z),
            'Near': ray.data.nearclip,
            'Far': ray.data.farclip,
            'X' : ray.data.x,
            'Y': ray.data.y,
            'Depth': get_depth(ray.data.x, ray.data.y),
            'Pos': (ray.data.position.x, ray.data.position.y, ray.data.position.z), 
            'VSB': vsb,
            'Axes': axes
        } for ray in RAYS 
    ]
    FLAGS.set_flag(FLAGS.RAYCASTS, False)
    return rays

def save_rays(rays):
    with open('rays.pickle', 'wb') as file:
        pickle.dump(rays, file)

def load_rays():
    with open('rays.pickle', 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':

    if '--plot' in sys.argv:
        R = load_rays()
        rays = R['Rays']
        proj = R['P']
        normalize = lambda arr: (np.array(arr) - np.array(arr).min()) / (np.array(arr).max() - np.array(arr).min())
        distance = [np.linalg.norm(np.array(ray['Ray']) - np.array(ray['Pos'])) for ray in rays]
        depth = [ray['Depth'] for ray in rays]
        far = [10003.815 for ray in rays]
        near = [0.15 for ray in rays]
        depth = [(-(f*n)/(n-f))/(d - (n/(n-f))) for d, n, f in zip(depth, near, far)]
        # plt.plot(distance, color='blue')
        # plt.plot(depth, color='red')
        # plt.plot(distance, color='blue')
        plt.plot(depth, color='red')
        plt.show(block=True)
        exit()
    else:
        print("Waiting for GTA V")
        FLAGS.wait_until(Flags.BEGIN_TRAINING, True)
        VideoState.init_cuda_arrays()
        print("VideoState Initialized")
        FLAGS.set_flag(Flags.REQUEST_GAME_STATE, True)
        FLAGS.set_flag(Flags.UNSTUCK, True)

while True:
    R = []
    while len(R) < 100:
        r = Rays()
        if len(R) == 0 or r[-1]['Pos'] != R[-1]['Pos']:
            R += r
    P = Proj()
    AR = VideoState.pop().shape[2] / VideoState.pop().shape[1]
    if '--save' in sys.argv:
        save_rays({'P': P, 'Rays': R, 'AR': AR})