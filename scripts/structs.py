import numpy as np
from ipc import Serializable
from typing import Optional

class Vec3f(metaclass=Serializable):
    x: np.float32
    y: np.float32
    z: np.float32

class RayCast(metaclass=Serializable):
    x: np.float32
    y: np.float32
    r: int
    g: int
    b: int
    a: int
    collision: Vec3f

class VertexShaderConstants(metaclass=Serializable):
    nearclip: np.float32
    farclip: np.float32
    constant_buffers: list[bytes]

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

class CUDAPitchedArrayObject(metaclass=Serializable):
    handle: bytes
    format: CUDAChannelFormatDesc
    pitch: np.uint64
    extent: CUDAExtent

class GameState(metaclass=Serializable):
    collided: bool
    reward: float

class KeyboardState(metaclass=Serializable):
    w: bool
    a: bool
    s: bool
    d: bool