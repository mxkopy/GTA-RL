from structs import *
from ipc import StructuredMemory

VSB = StructuredMemory("VSConstantBuffers")

def VSB():
    return np.frombuffer(VSB.data.constant_buffers[2], dtype=np.float32)

def Axes():
    vsb = VSB()
    X = vsb[28:31] - vsb[32:35]
    Y = vsb[40:43] - vsb[36:39]
    Z = vsb[16:19]
    return np.stack((X, Y, Z))

def Pos():
    vsb = VSB()
    return vsb[12:15]

def Misc():
    vsb = VSB()
    return {
        'VW': vsb[20],
        'VH': vsb[21],
        'SW': vsb[60],
        'SH': vsb[61],
        'Near': VSB.data.nearclip,
        'Far': VSB.data.farclip
    }

def RayFromNDC(x, y, z=1.0):
    A = Axes()
    M = Misc()
    x = x * (M['SW']/M['VW'])
    y = y * (M['SH']/M['VH'])
    return A.T @ np.array((x, y, z))
