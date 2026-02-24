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

# Global relevant for the serialization scheme
FMT_SIZE_T = '@N'
SIZEOF_SIZE_T = calcsize(FMT_SIZE_T)
IPC_SLEEP_DURATION = 0

# Whether or not to compile serialization schema
COMPILE = '--compile' in sys.argv

# Useful type hint functions to use when compiling flat/protobuffers 
def get_eltype(typehint):
    T = typehint
    while get_args(T) != ():
        T = get_args(typehint)[0]
    return T

def IS_LEAF(typehint):
    return get_args(typehint) == ()

def IS_OPTIONAL(typehint):
    return get_origin(typehint) is Union and type(None) in get_args(typehint) and len(get_args(typehint)) == 2

def IS_LIST(typehint):
    return get_origin(typehint) is list

def IS_TUPLE(typehint):
    return get_origin(typehint) is tuple

def IS_UNION(typehint):
    # Do not count Optional as also Union
    if IS_OPTIONAL(typehint):
        return False
    return get_origin(typehint) is Union

# Location of generated .proto files & protoc output
PROTO_DIR = 'protos'
FLAT_DIR = 'fbs'

# Compiler locations
PREFIX = 'E:\\GTA-RL\\dxinterop\\vcpkg_installed\\x64-windows-static-md\\tools'

PROTOC = f'{PREFIX}\\protobuf\\protoc.exe'
FLATC = f'{PREFIX}\\flatbuffers\\flatc.exe'

# String represenation of python types according to each scheme
PROTO_TYPES = {
    bool: 'bool',
    int: 'int64',
    float: 'double',
    bytes: 'bytes',
    str: 'string'
} | {
    np.int32: 'int32',
    np.uint32: 'uint32',
    np.uint64: 'uint64',
    np.float32: 'float',
}

FLAT_TYPES = {
    bool: 'bool',
    int: 'long',
    float: 'double',
    bytes: '[byte]',
    str: 'string'
} | {
    np.int32: 'int',
    np.uint32: 'uint32',
    np.uint64: 'ulong',
    np.float32: 'float'
}

# Parses a class attribute into a protobuf/flatbuf message field
def PROTO_FIELD(fieldname, fieldinfo):
    modifier = ' required'
    if fieldinfo['repeated']:
        modifier = f' repeated'
    elif fieldinfo['optional']:
        modifier = f' optional'
    return f"{modifier} {fieldinfo['typename']} {fieldname} = {fieldinfo['index']};"

def FLAT_FIELD(fieldname, fieldinfo):
    typename = fieldinfo['typename']
    if fieldinfo['repeated']:
        typename = f'[{typename}]'
    modifier = ''
    if not fieldinfo['optional']:
        modifier = f' (required)'
    return f'{fieldname}: {typename}{modifier};'

# Structure of a .proto/.fbs import
PROTO_INCLUDE = 'import "{clsname}.proto";'
FLAT_INCLUDE  = 'include "{clsname}";'

# Schema structure
PROTO_SCHEMA = "{includes}\nmessage {clsname} {{\n{field_strings}\n}}\n"
FLAT_SCHEMA = "{includes}\ntable {clsname} {{\n{field_strings}\n}}\nroot_type {clsname};"

# Python module imports naming convention
PROTO_PY_EXT = '{clsname}_pb2'
FLATC_PY_EXT = '{clsname}'

# Schema file extensions
PROTO_EXT = '{fname}.proto'
FLAT_EXT = '{fname}.fbs'

# Arguments to schema compilers
PROTOC_ARGS = [
    f'{PROTOC} --proto_path={PROTO_DIR} {PROTO_DIR}/{{clsname}}.proto --python_out={PROTO_DIR}',
    # There's a weird bug with MSVC & protobuffers where including multiple .cc files is not straightforward, since the
    # implementations reuse symbols in the global namespace. https://github.com/protocolbuffers/protobuf/issues/25457
    # The temporary solution is to independently compile an amalgamated proto file specifically for MSVC.
    f'{PROTOC} --proto_path={PROTO_DIR} {PROTO_DIR}/amalgamated.proto --cpp_out={PROTO_DIR}'
]

FLATC_ARGS = [
    f'{FLATC} --cpp --python -o {FLAT_DIR} -I {FLAT_DIR} {FLAT_DIR}/{{clsname}}.fbs'
]

# Schema-specific hooks
PROTOBUF_HOOKS = []

if '--flatbuf' in sys.argv:
    COMPILER = FLATC
    SCHEMA_DIR = FLAT_DIR
    TYPES = FLAT_TYPES
    FIELD = FLAT_FIELD
    INCLUDE = FLAT_INCLUDE
    SCHEMA = FLAT_SCHEMA
    SCHEMA_EXT = FLAT_EXT
    PY_EXT = FLATC_PY_EXT
    COMPILER_ARGS = FLATC_ARGS


else:
    COMPILER = PROTOC
    SCHEMA_DIR = PROTO_DIR
    TYPES = PROTO_TYPES
    FIELD = PROTO_FIELD
    INCLUDE = PROTO_INCLUDE
    SCHEMA = PROTO_SCHEMA
    SCHEMA_EXT = PROTO_EXT
    PY_EXT = PROTO_PY_EXT
    COMPILER_ARGS = PROTOC_ARGS


if COMPILE:
    shutil.rmtree(Path(SCHEMA_DIR), ignore_errors=True)
    os.mkdir(Path(SCHEMA_DIR))

sys.path.append(str(Path(Path.cwd(), SCHEMA_DIR)))


class Serializable(type):

    # Classes that use this metaclass
    REGISTERED_TYPES = []
    REGISTERED_TYPES_DICT = {}
    AMALGAMATED = ''

    def parse_field(fieldname, typehint, index):
        return PROTO_FIELD(fieldname, typehint, index)

    def parse_import(typehint):
        return INCLUDE(typehint)

    # TODO: dynamically construct new struct classes when the typehint is a tuple type
    def ast_node(cls, fieldname, typehint, index):
        if IS_TUPLE(typehint) or IS_UNION(typehint):
            raise NotImplementedError("Does not currently support tuple or union types.")
        T = get_eltype(typehint)
        return {
            'type': get_args(typehint) if IS_TUPLE(typehint) or IS_UNION(typehint) else T,
            'typename': T.__name__ if T not in TYPES else TYPES[T],
            'optional': IS_OPTIONAL(typehint),
            'repeated': IS_LIST(typehint),
            'tuple': IS_TUPLE(typehint),
            'union': IS_UNION(typehint),
            'default': None if fieldname not in cls.__dict__ else cls.__dict__[fieldname],
            'index': index
        }

    def parse_class_as_ast(cls, offset=1, keep_includes=True):
        ast = {}
        includes = set()
        fields = get_type_hints(cls).items()
        index = offset
        for fieldname, typehint in fields:
            node = Serializable.ast_node(cls, fieldname, typehint, index)
            ast[fieldname] = node
            if keep_includes and node['type'] in Serializable.REGISTERED_TYPES:
                includes.add(node['typename'])
            index += 1
        return ast, includes

    def parse_class(cls, keep_includes=True):
        ast, includes = Serializable.parse_class_as_ast(cls, keep_includes=keep_includes)
        include_strings = '\n'.join([INCLUDE.format(clsname=typename) for typename in includes])
        field_strings   = '\n'.join([FIELD(fieldname, ast[fieldname]) for fieldname in ast])
        schema = SCHEMA.format(includes=include_strings, clsname=cls.__name__, field_strings=field_strings)
        return schema

    def write_schema(cls):
        fname = SCHEMA_EXT.format(fname=cls.__name__)
        amalgamated_fname = SCHEMA_EXT.format(fname='amalgamated')
        schema = Serializable.parse_class(cls)
        Serializable.AMALGAMATED = f'{Serializable.AMALGAMATED}\n{Serializable.parse_class(cls, keep_includes=False)}'
        with open(Path(SCHEMA_DIR, fname), 'w') as file:
            file.write(schema)
        with open(Path(SCHEMA_DIR, amalgamated_fname), 'w') as file:
            file.write(Serializable.AMALGAMATED)

    def run_compiler(cls):
        for ARGSTR in COMPILER_ARGS:
            try:
                subprocess.run(
                    ARGSTR.format(clsname=cls.__name__), 
                    capture_output=True, 
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(e.stdout, e.stderr)
                raise e

    # Generate protostring and run protoc on class
    def compile(cls):
        Serializable.write_schema(cls)
        Serializable.run_compiler(cls)
 
    # Import generated .py file
    def import_schema_type(cls):
        module_name = PY_EXT.format(clsname=cls.__name__)
        module = importlib.import_module(module_name)
        return getattr(module, cls.__name__)

    # Patches some useful methods into the python protobuf class object  
    def patch_methods(MSG_TYPE):
        def init_hook(__init__original__):
            # If one positional argument is passed to __init__, assume it's a bytearray encoding the object 
            def __init__(self, *args, **kwargs):
                # __init__original__(self, **(cls.__defaults__ | kwargs))
                __init__original__(self, **kwargs)
                if len(args) == 1:
                    self.MergeFromString(*args)
            return __init__

        def __ixor__(self: Message, update: Message):
            self.MergeFrom(update)
            return self

        def __xor__(self: Message, update: Message):
            x = self.__class__()
            x.CopyFrom(self)
            x ^= update
            return x

        MSG_TYPE.__init__ = init_hook(MSG_TYPE.__init__)
        MSG_TYPE.__ixor__ = __ixor__
        MSG_TYPE.__xor__ = __xor__
        MSG_TYPE.frombuffer = MSG_TYPE.FromString
        MSG_TYPE.__call__ = MSG_TYPE.MergeFromString
        MSG_TYPE.__bytes__ = MSG_TYPE.SerializeToString

    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)
        if COMPILE:
            Serializable.compile(cls)
            # Serializable.cpp_compile(cls)
        MSG_TYPE = Serializable.import_schema_type(cls)
        Serializable.patch_methods(MSG_TYPE)
        Serializable.REGISTERED_TYPES.append(MSG_TYPE)
        Serializable.REGISTERED_TYPES_DICT[MSG_TYPE.__name__] = MSG_TYPE
        TYPES[MSG_TYPE] = MSG_TYPE.__name__
        return MSG_TYPE

    def __call__(cls):
        print(cls)

class ByteBuffer(metaclass=Serializable):
    data: bytes

class ByteBuffers(metaclass=Serializable):
    data: list[ByteBuffer]

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

    def flush(self):
        self.ipc.flush()

    def __init__(self, tagname):
        self.ipc = mmap.mmap(-1, 2*SIZEOF_SIZE_T + Memory.check_capacity(tagname), tagname)
        if self.capacity() == 0:
            self.change_capacity(1024)


# Get & set protobuf-encoded messages using the .data property
# Stores a string identifying the payload type along with the payload bytes
# Since we're in python, we can dynamically select which type to parse the bytes as
class StructuredMemory(Memory):

    class Payload(metaclass=Serializable):
        typename: str
        data: bytes

    @property
    def data(self) -> Message:
        payload = StructuredMemory.Payload(self.raw)
        T = Serializable.REGISTERED_TYPES_DICT[payload.typename]
        return T(payload.data)

    @data.setter
    def data(self, value: Message):
        self.payload.typename = type(value).__name__
        self.payload.data = bytes(value)
        self.raw = bytes(self.payload)

    def __init__(self, tagname, *args, **kwargs):
        super().__init__(tagname=tagname, *args, **kwargs)
        self.payload = StructuredMemory.Payload()

# Globally accessible synchronization flags 
class Flags:

    class FLAG(int):
        pass

    FLAGS_TAGNAME = "Flags"
    N_FLAGS = 2
    IPC_SLEEP_DURATION = 1e-3

    BEGIN_TRAINING: FLAG = 0
    REQUEST_GAME_STATE: FLAG = 1

    def __init__(self, n_flags=N_FLAGS, tagname=FLAGS_TAGNAME):
        self.flags = mmap.mmap(-1, -(n_flags // -8), tagname)

    def set_flag(self, idx: int, value: bool) -> None:
        pos, offset = idx // 8, idx % 8
        mask = ~(1 << offset)
        self.flags.seek(pos)
        state = self.flags.read_byte()
        self.flags.seek(pos)
        updated_state = (state & mask) | (value << offset)
        self.flags.write_byte(updated_state)
        self.flags.flush()

    def get_flag(self, idx: int) -> bool:
        pos, offset = idx // 8, idx % 8
        mask = 1 << offset
        self.flags.seek(pos)
        state = self.flags.read_byte()
        return (state & mask) != 0

    def wait_until(self, idx: int, value: bool, fn = lambda: time.sleep(IPC_SLEEP_DURATION)) -> None:
        while self.get_flag(idx) != value:
            fn()

    @staticmethod
    def debug(flag=None, value=None):
        flags = Flags()
        if flag is not None:
            if value is None:
                flags.set_flag(flag, not flags.get_flag(flag))
            else:
                flags.set_flag(flag, value)
            return
        for fieldname, typehint in get_type_hints(Flags).items():
            print(f'{fieldname}: {flags.get_flag(getattr(Flags, fieldname))}')


# Memory with a specific synchronization pattern.
# By default, assume all flags are set to 0.
#  
# When requesting game data: 
#   set the request flag to 1. 
#   wait until the flag is 0
#   read memory section & return.
# 
# When producing game data:
#   wait until the request flag is set to 1. 
#   write data to memory section. 
#   set request flag to 0 & return
#  
class RequestLockedMemory(StructuredMemory):

    def __init__(self, tagname, request_signal = Flags.REQUEST_GAME_STATE, *args, **kwargs):
        super().__init__(tagname, *args, **kwargs)
        self.flags = Flags()
        self.request_signal = request_signal
    
    @property
    def data(self) -> Message:
        self.flags.set_flag(self.request_signal, True)
        self.flags.wait_until(self.request_signal, False)
        return StructuredMemory.data.fget(self)

    @data.setter
    def data(self, msg: Message):
        self.flags.wait_until(self.request_signal, True)
        StructuredMemory.data.fset(self, msg)
        self.flags.set_flag(self.request_signal, False)

if __name__ == '__main__':

    from structs import Vec3f, GameState
    from structs import Serializable

    if '--vsb' in sys.argv:
        import numpy as np
        VSB = StructuredMemory("VSConstantBuffers")
        while True:
            vsb = VSB.data.constant_buffers[2]
            vsb = np.frombuffer(vsb, dtype=np.float32).reshape(-1, 4)
            print(vsb)

    if '--test' in sys.argv:
        reader = RequestLockedMemory("GameState")
        while True:
            reader.data

    if '--start' in sys.argv:
        Flags().set_flag(Flags.REQUEST_GAME_STATE, True)
        exit()

    if '--writer' in sys.argv:
        writer = RequestLockedMemory("Test")
        while True:
            x, y, z = np.random.rand(3)
            test = Vec3f(x=x, y=y, z=z)
            print(f'Writing data {x} {y} {z}')
            writer.data = test 

    if '--reader' in sys.argv:
        reader = RequestLockedMemory("Test")
        while True:
            print('Reading data')
            print(reader.data)

