import mmap
from typing import Iterable
import time
import msgs_pb2
from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor


class Mutex:

    def __init__(self, name=u"RLLibWorkerMutex"):
        # Windows Mutex
        import ctypes
        kernel32 = ctypes.windll.kernel32
        self.CreateMutexW = kernel32.CreateMutexW
        self.WaitForSingleObject = kernel32.WaitForSingleObject
        self.ReleaseMutex = kernel32.ReleaseMutex
        self.CloseHandle = kernel32.CloseHandle
        self.Mutex = self.CreateMutexW(None, False, name)

    def acquire(self, timeout=0xFFFFFFFF):
        acquired = self.WaitForSingleObject(self.Mutex, timeout)
        return acquired == 0

    def release(self):
        self.ReleaseMutex(self.Mutex)

    def close(self):
        self.CloseHandle(self.Mutex)

N_FLAGS = 2
FLAGS_TAG = "flags.ipc"
IPC_SLEEP_DURATION = 1e-3
    
class FLAGS:

    BEGIN_TRAINING = 0
    REQUEST_GAME_STATE = 1



class Flags:

    def __init__(self, n_flags=N_FLAGS, tagname=FLAGS_TAG):
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

    def wait_until(self, idx: int | Iterable[int], value: bool | Iterable[bool], fn = lambda: time.sleep(IPC_SLEEP_DURATION)) -> None:
        if isinstance(idx, int) and isinstance(value, bool):
            while self.get_flag(idx) != value:
                fn()
        else:
            while sum(self.get_flag(i) != v for i, v in zip(idx, value)) > 0:
                fn()

class Channel:

    def __init__(self, size: int, tagname: str):
        self.ipc = mmap.mmap(-1, size, tagname)

    def close(self) -> None:
        self.ipc.flush()
        self.ipc.close()

    def push_nbl(self, payload: bytes, offset=0) -> None:
        self.ipc.seek(offset)
        self.ipc.write(payload)
        self.ipc.flush()

    def pop_nbl(self, offset=0, numbytes=-1) -> bytes:
        self.ipc.seek(offset)
        payload: bytes = self.ipc.read(numbytes)
        return payload

class MessageMapped(type):

    def default_init_dict(descriptor: FieldDescriptor):
        descriptor.fields: list[FieldDescriptor]
        return {
            field.name: 0 if field.message_type is None else MessageMapped.default_init_dict(field.message_type)
            for field in descriptor.fields
        }

    def create_init(msg_type: type[Message]):
        def init():
            return msg_type(**MessageMapped.default_init_dict(msg_type.DESCRIPTOR))
        return init

    def post_init_hook(cls_init):
        def hook(self, *args, **kwargs):
            cls_init(self, *args, **kwargs)
            self.push_nbl(self.MSG_TYPE.init())
        return hook

    def from_iterable(message: Message, itr: Iterable):
        for field, x in zip(message.DESCRIPTOR.fields, itr):
            value = getattr(message, field.name)
            if not isinstance(value, Message) and x is not None:
                setattr(message, field.name, x)
            elif x is not None:
                MessageMapped.from_iterable(value, x)
        return message

    def to_list(message: Message):
        return [getattr(message, field.name) if field.message_type is None else MessageMapped.to_list(getattr(message, field.name)) for field in message.DESCRIPTOR.fields]

    def to_tuple(message: Message):
        return tuple(getattr(message, field.name) if field.message_type is None else MessageMapped.to_tuple(getattr(message, field.name)) for field in message.DESCRIPTOR.fields)

    def push_nbl(self, payload: Message):
        assert payload.__class__ == self.__class__.MSG_TYPE
        msg: bytes = payload.SerializeToString()
        Channel.push_nbl(self, msg)

    def pop_nbl(self) -> Message:
        msg: bytes = Channel.pop_nbl(self)
        return self.__class__.MSG_TYPE.FromString(msg)

    def __xor__(self, update: Message):
        current_state: Message = self.pop_nbl()
        current_state.MergeFrom(update)
        return current_state
    
    def __ixor__(self, update: Message):
        updated_state = self ^ update
        self.push_nbl(updated_state)
        if isinstance(self, Queue):
            self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, True)
        return self

    def __init__(cls, *args, **kwargs):
        # assert cls subtype of Channel and Flags
        super().__init__(*args, **kwargs)
        cls.MSG_TYPE.init = MessageMapped.create_init(cls.MSG_TYPE)
        cls.N_BYTES = cls.MSG_TYPE(**MessageMapped.default_init_dict(cls.MSG_TYPE.DESCRIPTOR)).ByteSize()
        cls.MSG_TYPE.to_list = MessageMapped.to_list
        cls.MSG_TYPE.to_tuple = MessageMapped.to_tuple
        cls.MSG_TYPE.from_iterable = MessageMapped.from_iterable
        cls.pop_nbl = MessageMapped.pop_nbl
        cls.push_nbl = MessageMapped.push_nbl
        cls.__xor__ = MessageMapped.__xor__
        cls.__ixor__ = MessageMapped.__ixor__
        cls.__init__ = MessageMapped.post_init_hook(cls.__init__)

class Queue(Flags, Channel):

    LOCK: FLAGS
    TAGNAME: str

    def push(self, state: tuple | list | Message):
        self.wait_until(self.__class__.LOCK, True)
        self.push_nbl(state)
        self.set_flag(self.__class__.LOCK, False)

    def pop(self) -> tuple:
        self.set_flag(self.__class__.LOCK, True)
        self.wait_until(self.__class__.LOCK, False)
        state = self.pop_nbl()
        return state

    def __init__(self, size: int, tagname: str):
        Flags.__init__(self)
        Channel.__init__(self, size, tagname)

# class StateQueue(Flags, metaclass=MessageChannel):

#     READY_TO_READ: int
#     NEW_MESSAGE_WRITTEN: int
#     TAGNAME: str

#     N_BYTES: int
#     MSG_TYPE: type[Message]

#     def __init__(self):
#         Flags.__init__(self)        
#         MappedChannel.__init__(self, tagname=self.__class__.TAGNAME)

#     def push(self, state: tuple | list | Message):
#         self.wait_until(self.__class__.READY_TO_READ, True)
#         self.push_nbl(state)
#         self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, True)

#     def pop(self) -> tuple:
#         self.set_flag(self.__class__.READY_TO_READ, True)
#         self.wait_until(self.__class__.NEW_MESSAGE_WRITTEN, True)
#         state: Message = self.pop_nbl()
#         self.set_flag(self.__class__.READY_TO_READ, False)
#         self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, False)
#         return state
    
#     def __xor__(self, update: Message):
#         current_state: Message = self.pop_nbl()
#         current_state.MergeFrom(update)
#         return current_state
    
#     def __ixor__(self, update: Message):
#         updated_state = self ^ update
#         self.push_nbl(updated_state)
#         self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, True)
#         return self

def debug_flags():
    flags = Flags()
    print(f'BEGIN_TRAINING: {flags.get_flag(FLAGS.BEGIN_TRAINING)}')
    print(f'REQUEST_GAME_STATE: {flags.get_flag(FLAGS.REQUEST_GAME_STATE)}')

if __name__ == '__main__':
    debug_flags()