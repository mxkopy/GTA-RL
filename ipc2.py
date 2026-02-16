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
from struct import pack, unpack, calcsize
from pathlib import Path
from typing import Literal, Union, Tuple, Dict, Optional, get_args, get_origin, get_type_hints
from pprint import pp
from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor


# At this stage I need a much more principled way of dealing with game data so I can quickly write scripts that use it
# The idea is this:
# Create central registry for all allocated IPC memory metadata 
# Metadata should include the memory address, device, and other relevant information (shape, dtype, etc) as a protobuffer
# On top of registry, should provide a 'retriever' class that provides a view of the data

FMT_SIZE_T = '@N'
SIZEOF_SIZE_T = calcsize(FMT_SIZE_T)
REGISTRY_FILENAME="ipcdata.registry"
IPC_SLEEP_DURATION = 0.1

Path(REGISTRY_FILENAME).unlink(missing_ok=True)


# Location of generated .proto files & protoc output
PROTO_DIR = 'protos'
shutil.rmtree(Path(PROTO_DIR))
os.mkdir(Path(PROTO_DIR))

sys.path.append(str(Path(Path.cwd(), PROTO_DIR)))

class Serializable(type):

    TYPES = {
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

    # Classes that use this metaclass
    REGISTERED_TYPES = set()

    # Gets the (first, assumed only for now) leaf type of a type hint
    def get_eltype(type_hint):
        T = type_hint
        while get_args(T) != ():
            T = get_args(type_hint)[0]
        return T

    # Parses a class attribute into a protobuf message field
    def parse_field(fieldname, type_hint, index):
        T = Serializable.get_eltype(type_hint)
        modifier = 'required'
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
    def parse_class(cls):
        imports = []
        message_fields = []
        for i, (fieldname, type_hint) in enumerate(get_type_hints(cls).items(), start=1):
            imports += [Serializable.parse_import(type_hint)] if Serializable.parse_import(type_hint) is not None else []
            message_fields += [Serializable.parse_field(fieldname, type_hint, i)]
        importstring = '\n'.join(imports)
        fieldstring = '\n'.join(message_fields)
        messagestring = f'message {cls.__name__} {{\n{fieldstring}\n}}'
        return f'{importstring}\n{messagestring}'

    # Records default values for class members 
    def parse_defaults(cls):
        cls.__defaults__ = {}
        for fieldname in get_type_hints(cls):
            if fieldname in cls.__dict__:
                cls.__defaults__[fieldname] = cls.__dict__[fieldname]

    # Runs protoc on generated protostring
    def compile_protostring(name, protostring: str):
        fname = f'{name}.proto'
        with open(Path(PROTO_DIR, fname), 'w') as file:
            file.write(protostring)
        import subprocess
        try:
            subprocess.run(
                ['protoc', f'--proto_path={PROTO_DIR}', f'{PROTO_DIR}/*.proto', f'--cpp_out={PROTO_DIR}', f'--python_out={PROTO_DIR}'], 
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

    def cpp_compile_includes():
        includes = []
        for T in Serializable.REGISTERED_TYPES:
            includes.append(f'#include "{T.__name__}.pb.h"')
        return '\n'.join(includes)

    def cpp_compile_cases(cls):
        pass

    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)
        Serializable.compile_protostring(cls.__name__, Serializable.parse_class(cls))
        Serializable.parse_defaults(cls)
        Serializable.assign_messagetype(cls)
        Serializable.patch_methods(cls)
        Serializable.TYPES[cls.MSG_TYPE] = cls.MSG_TYPE.__name__
        Serializable.REGISTERED_TYPES.add(cls.MSG_TYPE)
        with open(Path(PROTO_DIR, 'all_protos.h'), 'w') as file:
            cpp_includes = Serializable.cpp_compile_includes()
            file.write(cpp_includes)
        return cls.MSG_TYPE

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

class Registered(type):

    class Registry(metaclass=Serializable):
        registry: list[str]

    # REGISTRY = StructuredMemory('registry')

    def __new__(mcls, name, bases, dct):
        return Serializable.__new__(mcls, name, bases, dct)


class Dataview(type):

    ptr: bytes | str
    metadata: type[Message]

    # def __init__(mcls, name, bases, dct):        
    #     cls = super().__new__(*args, **kwargs)
