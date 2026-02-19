import mmap
import time
import threading
import shelve
import collections
import importlib
import numpy as np
import os
import sys
import shutil
import subprocess
from struct import pack, unpack, calcsize
from pathlib import Path
from typing import Literal, Union, Tuple, Dict, Optional, get_args, get_origin, get_type_hints
from pprint import pp
from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor

# At this stage I need a much more principled way of dealing with game data.
# The idea is this:
# Create central registry for all allocated IPC memory metadata 
# Metadata should include the memory address, device, and other relevant information (shape, dtype, etc) as a protobuffer
# On top of registry, should provide a 'retriever' class that provides a view of the data

FMT_SIZE_T = '@N'
SIZEOF_SIZE_T = calcsize(FMT_SIZE_T)
IPC_SLEEP_DURATION = 0.1

# Useful type hint functions to use when compiling flat/protobuffers 
def get_eltype(type_hint):
    T = type_hint
    while get_args(T) != ():
        T = get_args(type_hint)[0]
    return T

def IS_OPTIONAL(type_hint):
    T_args = get_args(type_hint)
    return len(T_args) == 2 and T_args[1] == None

def IS_LIST(type_hint):
    return get_origin(type_hint) is list

# Location of generated .proto files & protoc output
PROTO_DIR = 'protos'
FLAT_DIR = 'fbs'

# Compiler locations
PROTOC = 'E:\\GTA-RL\\dxinterop\\vcpkg_installed\\vcpkg\\pkgs\\protobuf_x64-windows\\tools\\protobuf\\protoc.exe'

# String represenation of python types according to each scheme
PROTO_TYPES = {
    bool: 'bool',
    bytes: 'bytes',
    str: 'string',
    float: 'double',
    int: 'int64'
} | {
    np.float32: 'float',
    np.int32: 'int32',
    np.uint32: 'uint32',
    np.uint64: 'uint64'
}

FLAT_TYPES = PROTO_TYPES

# Field syntax for each scheme
def PROTO_FIELD(name, type_hint, index):
    modifier = ''
    T_str = PROTO_TYPES[get_eltype(type_hint)]
    if IS_LIST(type_hint):
        modifier = f' repeated{modifier}'
    if IS_OPTIONAL(type_hint):
        modifier = f' optional{modifier}'
    return f'{modifier} {T_str} {name} = {index};'

def FLAT_FIELD(name, type_hint, *args):
    T_str = FLAT_TYPES[get_eltype(type_hint)]
    if IS_LIST(type_hint):
        T_str = f'[{T_str}]'
    modifier = ''
    if not IS_OPTIONAL(type_hint):
        modifier = f' (required)'
    return f'{name}: {T_str}{modifier};'

# Object structure for each scheme
def PROTO_OBJECT(name: str, fieldstr: str):
    return f'message {name} {{\n{fieldstr}\n}}\n'

def FLAT_OBJECT(name, fieldstr: str):
    return f'table {name} {{\n{fieldstr}\n}}\n'

# Import syntax for each scheme
def PROTO_IMPORT(T):
    return f'import "{T.__name__}";'

def FLAT_IMPORT(T):
    return f'include "{T.__name__}";'

def remove_and_recreate(dir):
    shutil.rmtree(Path(dir), ignore_errors=True)
    os.mkdir(Path(dir))
    sys.path.append(str(Path(Path.cwd(), dir)))

remove_and_recreate(PROTO_DIR)
remove_and_recreate(FLAT_DIR)


class Serializable(type):

    TYPES = PROTO_TYPES

    # Classes that use this metaclass
    REGISTERED_TYPES = []

    # Gets the (first, assumed only for now) leaf type of a type hint
    def get_eltype(type_hint):
        T = type_hint
        while get_args(T) != ():
            T = get_args(type_hint)[0]
        return T

    # Parses a class attribute into a protobuf message field
    def parse_field(fieldname, type_hint, index):
        # T = get_eltype(type_hint)
        # return PROTO_FIELD(fieldname, type_hint, index)
        T = get_eltype(type_hint)
        if get_origin(type_hint) is list:
            modifier = 'repeated'
        else:
            modifier = 'optional'
        return f'{modifier} {Serializable.TYPES[T]} {fieldname} = {index};'

    # Parses a type hint into a .proto import 
    def parse_import(type_hint):
        T = Serializable.get_eltype(type_hint)
        if T in Serializable.REGISTERED_TYPES:
            return f'import "{T.__name__}.proto";'

    # Parses a class into a protobuf message type
    def parse_class(cls, incl_imports=True):
        imports = []
        message_fields = []
        cls_fields = get_type_hints(cls).items()
        for i, (fieldname, type_hint) in enumerate(cls_fields, start=1):
            imports += [Serializable.parse_import(type_hint)] if Serializable.parse_import(type_hint) is not None else []
            message_fields += [Serializable.parse_field(fieldname, type_hint, i)]
        importstring = '\n'.join(imports)
        fieldstring = '\n'.join(message_fields)
        messagestring = f'message {cls.__name__} {{\n{fieldstring}\n}}'
        return f'{importstring if incl_imports else ""}\n{messagestring}'

    # Records default values for class members 
    def parse_defaults(cls):
        cls.__defaults__ = {}
        for fieldname in get_type_hints(cls):
            if fieldname in cls.__dict__:
                cls.__defaults__[fieldname] = cls.__dict__[fieldname]

    # Generate protostring and run protoc on class
    def compile(cls):
        fname = f'{cls.__name__}.proto'
        protostring = Serializable.parse_class(cls)
        with open(Path(PROTO_DIR, fname), 'w') as file:
            file.write(protostring)
        try:
            subprocess.run(
                [PROTOC, f'--proto_path={PROTO_DIR}', f'{PROTO_DIR}/{cls.__name__}.proto', f'--python_out={PROTO_DIR}'], 
                capture_output=True, 
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(e.stdout, e.stderr)
            raise e

    # There's a weird bug with MSVC & protobuffers where including multiple .cc files is not straightforward, since the
    # implementations reuse symbols in the global namespace. https://github.com/protocolbuffers/protobuf/issues/25457
    # The temporary solution is to independently compile a unified proto file specifically for MSVC. 
    def cpp_compile(cls):
        if not hasattr(Serializable, 'cpp_protostring'):
            Serializable.cpp_protostring = ''
        fname = f'cpp.proto'
        Serializable.cpp_protostring = f'{Serializable.cpp_protostring}\n{Serializable.parse_class(cls, incl_imports=False)}'
        with open(Path(PROTO_DIR, fname), 'w') as file:
            file.write(Serializable.cpp_protostring)
        try:
            subprocess.run(
                [PROTOC, f'--proto_path={PROTO_DIR}', f'{PROTO_DIR}/cpp.proto', f'--cpp_out={PROTO_DIR}'], 
                capture_output=True, 
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(e.stdout, e.stderr)
            raise e

    # Assign protobuf type to class
    def assign_messagetype(cls):
        module_name = f'{cls.__name__}_pb2'
        module = importlib.import_module(module_name)
        cls.MSG_TYPE = getattr(module, cls.__name__)

    # Patches some useful methods into the python protobuf class object  
    def patch_methods(cls):
        def init_hook(__init__original__):
            def __init__(self, *args, **kwargs):
                # __init__original__(self, **(cls.__defaults__ | kwargs))
                __init__original__(self, **kwargs)
                if len(args) != 0:
                    self.MergeFrom(self.FromString(*args))
            return __init__

        def __ixor__(self: Message, update: Message):
            self.MergeFrom(update)
            return self

        def __xor__(self: Message, update: Message):
            x = self.__class__()
            x.CopyFrom(self)
            x ^= update
            return x

        cls.MSG_TYPE.__init__ = init_hook(cls.MSG_TYPE.__init__)
        cls.MSG_TYPE.__ixor__ = __ixor__
        cls.MSG_TYPE.__xor__ = __xor__
        cls.MSG_TYPE.frombuffer = cls.MSG_TYPE.FromString
        cls.MSG_TYPE.__call__ = cls.MSG_TYPE.FromString
        cls.MSG_TYPE.__bytes__ = cls.MSG_TYPE.SerializeToString

    def cpp_compile_includes(ext: Literal['cc'] | Literal['h']):
        includes = []
        for T in Serializable.REGISTERED_TYPES:
            includes.append(f'#include "{T.__name__}.pb.{ext}"')
        return '\n'.join(includes)

    def cpp_compile_cases():
        cases = []
        for T in Serializable.REGISTERED_TYPES:
            cases.append(f'\tif constexpr(std::is_same_v<T, {T.__name__}>) return std::string("{T.__name__}");')
        return (
            f"template<typename T>\n"
            f"std::string ProtoTypeName()\n"
            f"{{\n"
            f"{'\n'.join(cases)}\n"
            f"}}\n"
        )

    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)
        Serializable.compile(cls)
        Serializable.cpp_compile(cls)
        Serializable.parse_defaults(cls)
        Serializable.assign_messagetype(cls)
        Serializable.patch_methods(cls)
        Serializable.TYPES[cls.MSG_TYPE] = cls.MSG_TYPE.__name__
        Serializable.REGISTERED_TYPES.append(cls.MSG_TYPE)
        # for ext in ['cc', 'h']:
            # with open(Path(PROTO_DIR, f'all_protos.{ext}'), 'w') as file:
                # includes = Serializable.cpp_compile_includes(ext)
                # file.write(includes)
        return cls.MSG_TYPE

    def __call__(cls):
        print(cls)


class ByteBuffers(metaclass=Serializable):
    data: list[bytes]

def pack_size_t(size_t: int):
    return pack(FMT_SIZE_T, size_t)

def unpack_size_t(buffer: bytes) -> int:
    return unpack(FMT_SIZE_T, buffer)[0]

# First 8 bytes are dedicated to the capacity of of the segment
# Next 8 bytes are dedicated to the current length of the segment
# Essentially a resizable array type thing
class Memory:

    @staticmethod
    def check_capacity(name: str) -> int:
        test = mmap.mmap(-1, SIZEOF_SIZE_T, tagname=name)
        cap = unpack_size_t(test[0:SIZEOF_SIZE_T])
        test.close()
        return cap

    @staticmethod
    def exists(name: str) -> bool:
        return Memory.check_capacity(name) > 0

    def change_capacity(self, new_capacity):
        self.ipc.resize(2*SIZEOF_SIZE_T + new_capacity)
        self.ipc[:SIZEOF_SIZE_T] = pack_size_t(new_capacity)

    def capacity(self) -> int:
        return unpack_size_t(self.ipc[:SIZEOF_SIZE_T])

    def __len__(self) -> int:
        return unpack_size_t(self.ipc[SIZEOF_SIZE_T:2*SIZEOF_SIZE_T])

    @property
    def raw(self) -> bytes:
        return self.ipc[2*SIZEOF_SIZE_T:2*SIZEOF_SIZE_T+len(self)]
    
    @raw.setter
    def raw(self, data: bytes):
        while len(data) >= self.capacity():
            self.change_capacity(2 * self.capacity())
        self.ipc[SIZEOF_SIZE_T:2*SIZEOF_SIZE_T] = pack_size_t(len(data))
        self.ipc[2*SIZEOF_SIZE_T:2*SIZEOF_SIZE_T+len(data)] = data

    def __init__(self, tagname):
        self.ipc = mmap.mmap(-1, 2*SIZEOF_SIZE_T + Memory.check_capacity(tagname), tagname)
        if self.capacity() == 0:
            self.change_capacity(1)

# Stores a string identifying the payload type along with the payload bytes
# Since we're in python, we can dynamically select which type to parse the bytes as
class StructuredMemory(Memory):

    class Payload(metaclass=Serializable):
        typename: str
        data: bytes

    @property
    def data(self):
        self.payload(self.raw)
        T = getattr(sys.modules[__name__], self.payload.typename)
        return T(self.payload.data)

    @data.setter
    def data(self, value: Message):
        self.payload.typename = type(value).__name__
        self.payload.data = bytes(value)
        self.raw = bytes(self.payload)

    def __init__(self, tagname, *args, **kwargs):
        super().__init__(tagname=tagname, *args, **kwargs)
        self.payload = StructuredMemory.Payload(self.raw)

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
    formatDesc: CUDAChannelFormatDesc
    bpp: np.uint64
    pitch: np.uint64
    extent: CUDAExtent

if __name__ == '__main__':

    SM = StructuredMemory("TEST")
    X = Vec3f(x=1, y=2, z=3)
    SM.data = X
    print(SM.data)
