import mmap
from typing import Iterable
import time
import msgs_pb2
from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor

N_FLAGS = 5
FLAGS_TAG = "flags.ipc"
IPC_SLEEP_DURATION = 1e-3
    
class FLAGS:

    REQUEST_GAME_STATE = 0
    REQUEST_ACTION = 1

    GAME_STATE_WRITTEN = 2
    ACTION_WRITTEN = 3

    RESET = 4
    IS_TRAINING = 5


class Flags:

    def __init__(self):
        self.flags = mmap.mmap(-1, -(N_FLAGS // -8), FLAGS_TAG)

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

    def push_nbl(self, payload: bytes) -> None:
        self.ipc.seek(0)
        self.ipc.write(payload)
        self.ipc.flush()

    def pop_nbl(self) -> bytes:
        self.ipc.seek(0)
        payload: bytes = self.ipc.read(-1)
        return payload

class MessageChannel(type):

    def default_init_dict(descriptor: FieldDescriptor):
        descriptor.fields: list[FieldDescriptor]
        return {
            field.name: 0 if field.message_type is None else MessageChannel.default_init_dict(field.message_type)
            for field in descriptor.fields
        }

    def create_init(msg_type: type[Message]):
        def init():
            return msg_type(**MessageChannel.default_init_dict(msg_type.DESCRIPTOR))
        return init

    def from_iterable(message: Message, itr: Iterable):
        for field, x in zip(message.DESCRIPTOR.fields, itr):
            value = getattr(message, field.name)
            if not isinstance(value, Message) and x is not None:
                setattr(message, field.name, x)
            elif x is not None:
                MessageChannel.from_iterable(value, x)
        return message

    def to_list(message: Message):
        return [getattr(message, field.name) if field.message_type is None else MessageChannel.to_list(getattr(message, field.name)) for field in message.DESCRIPTOR.fields]

    def to_tuple(message: Message):
        return tuple(getattr(message, field.name) if field.message_type is None else MessageChannel.to_tuple(getattr(message, field.name)) for field in message.DESCRIPTOR.fields)

    def push_nbl(self, payload: Message):
        assert payload.__class__ == self.__class__.MSG_TYPE
        msg: bytes = payload.SerializeToString()
        Channel.push_nbl(self, msg)

    def pop_nbl(self) -> Message:
        msg: bytes = Channel.pop_nbl(self)
        return self.__class__.MSG_TYPE.FromString(msg)

    # def __new__(mcls, name, bases, dict):
        # cls = type(name, bases, dict)
        # cls.push_nbl = MessageChannel.push_nbl
        # cls.pop_nbl = MessageChannel.pop_nbl
        # return cls

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.MSG_TYPE.init = MessageChannel.create_init(cls.MSG_TYPE)
        cls.N_BYTES = cls.MSG_TYPE(**MessageChannel.default_init_dict(cls.MSG_TYPE.DESCRIPTOR)).ByteSize()
        cls.MSG_TYPE.to_list = MessageChannel.to_list
        cls.MSG_TYPE.to_tuple = MessageChannel.to_tuple
        cls.MSG_TYPE.from_iterable = MessageChannel.from_iterable
        cls.pop_nbl = MessageChannel.pop_nbl
        cls.push_nbl = MessageChannel.push_nbl

class MessageQueue(MessageChannel):

    def push(self, state: tuple | list | Message):
        self.wait_until(self.__class__.READY_TO_READ, True)
        self.push_nbl(state)
        self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, True)

    def pop(self) -> tuple:
        self.set_flag(self.__class__.READY_TO_READ, True)
        # self.wait_until(self.__class__.NEW_MESSAGE_WRITTEN, True)
        state: Message = self.pop_nbl()
        self.set_flag(self.__class__.READY_TO_READ, False)
        self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, False)
        return state
    
    def __xor__(self, update: Message):
        current_state: Message = self.pop_nbl()
        current_state.MergeFrom(update)
        return current_state
    
    def __ixor__(self, update: Message):
        updated_state = self ^ update
        self.push_nbl(updated_state)
        self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, True)
        return self

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cls.pop = MessageQueue.pop
        # cls.push = MessageQueue.push
        # cls.__xor__ = MessageQueue.__xor__
        # cls.__ixor__ = MessageQueue.__ixor__

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
    print(f'REQUEST_GAME_STATE: {flags.get_flag(FLAGS.REQUEST_GAME_STATE)}')
    print(f'REQUEST_ACTION: {flags.get_flag(FLAGS.REQUEST_ACTION)}')
    print(f'GAME_STATE_WRITTEN: {flags.get_flag(FLAGS.GAME_STATE_WRITTEN)}')
    print(f'ACTION_WRITTEN: {flags.get_flag(FLAGS.ACTION_WRITTEN)}')
    print(f'RESET: {flags.get_flag(FLAGS.RESET)}')
    print(f'IS_TRAINING: {flags.get_flag(FLAGS.IS_TRAINING)}')
    flags.flags.seek(0)
    print(f'{flags.flags.read_byte()}')

if __name__ == '__main__':
    debug_flags()