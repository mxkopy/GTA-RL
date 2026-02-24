import numpy as np
from ipc import Serializable

class Vec3f(metaclass=Serializable):
    x: np.float32
    y: np.float32
    z: np.float32

class CUDAExtent(metaclass=Serializable):
    width: int
    height: int
    depth: int

class CUDAChannelFormatDesc(metaclass=Serializable):
    x: np.uint32
    y: np.uint32
    z: np.uint32
    w: np.uint32
    f: np.uint32

class CUDAArrayObject(metaclass=Serializable):
    handle: bytes
    format: CUDAChannelFormatDesc
    bpp: np.uint64
    pitch: np.uint64
    extent: CUDAExtent

class VertexShaderConstants(metaclass=Serializable):
    nearclip: np.float32
    farclip: np.float32
    constant_buffers: list[bytes]

class GameState(metaclass=Serializable):
    camera_direction: Vec3f
    velocity: Vec3f
    collided: bool
