import torch
import random
import config
import torchvision
import math
import mmap
import cupy
import numpy as np
from msgs_pb2 import ControllerState
from google.protobuf.message import Message
from collections import namedtuple
from typing import Iterable, TypeVarTuple, TypeAlias
from struct import unpack
from ipc import Flags, FLAGS, Channel, MessageQueue
from gymnasium import Env
from gymnasium.spaces import Tuple, Box, Discrete

import msgs_pb2

class GameState(Flags, Channel, metaclass=MessageQueue):

    READY_TO_READ = FLAGS.REQUEST_GAME_STATE
    NEW_MESSAGE_WRITTEN = FLAGS.GAME_STATE_WRITTEN
    TAGNAME = 'game_state.ipc'
    MSG_TYPE = msgs_pb2.GameState

    def __init__(self):
        Flags.__init__(self)
        Channel.__init__(self, GameState.N_BYTES, GameState.TAGNAME)

    def pop(self):
        game_state = MessageQueue.pop(self)
        return (
            (game_state.velocity.x, game_state.velocity.y, game_state.velocity.z),
            game_state.damage
        )

class VideoState:

    cuda_arrays = {}
    tensors = {}
    nearclipfarclip = Channel(8, "NearClipFarClip")

    def __init__(self, queue_length=100, depth=True):
        self.depth = depth

    def get_dtype(components, bpp):
        if bpp == 4 and components == 4:
            return cupy.uint8
        if bpp == 4 and components == 1:
            return cupy.float32

    def init_cuda_array(id):
        if id not in VideoState.tensors:
            try: 
                from ipc import Channel
                array_handle = Channel(64, f"{id}")
                array_format = Channel(32, f"{id}Info")
                memory_handle_bytes = array_handle.pop_nbl()
                components, bpp, pitch, height = unpack("@4P", array_format.pop_nbl())
                memory_handle = cupy.cuda.runtime.ipcOpenMemHandle(memory_handle_bytes)
                mem_buffer = cupy.cuda.UnownedMemory(memory_handle, pitch * height, owner=VideoState, device_id=0)
                memory_pointer = cupy.cuda.MemoryPointer(mem_buffer, 0)
                dtype = VideoState.get_dtype(components, bpp)
                cuda_array = cupy.ndarray(shape=(height, pitch // bpp, components), dtype=dtype, memptr=memory_pointer)
                VideoState.cuda_arrays[id] = cuda_array
                VideoState.tensors[id] = torch.from_dlpack(cuda_array)
            except Exception as exception:
                exception.add_note(f"Error occurred while initializing {id} in VideoState")
                print(exception)
                exit()

    def rescale(img: torch.Tensor):
        return torch.nn.functional.interpolate(img, config.state_sizes['image'], mode='bilinear', antialias=True)

    def pop_rgb():
        VideoState.init_cuda_array("RGB")
        img = VideoState.tensors["RGB"]
        img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16)
        img = img / 255
        img = VideoState.rescale(img) 
        return img

    def linearize_depth(array):
        near, far = unpack('@2f', VideoState.nearclipfarclip.pop_nbl())
        y = (torch.pow(far/near,array)-1) * (near / far)
        depth = ((y * near) / far)
        return depth

    def pop_depth():
        VideoState.init_cuda_array("DepthBuffer")
        depth = VideoState.linearize_depth(VideoState.tensors["DepthBuffer"])
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        depth = VideoState.rescale(depth).squeeze()
        return depth

    def pop(self) -> torch.Tensor:
        if self.depth:
            img = VideoState.pop_depth().cpu()
        else:
            img = img[:, :, :3].permute(2, 0, 1).unsqueeze(0)
            if self.grayscale:
                img = torchvision.transforms.functional.rgb_to_grayscale(img)
            img = img.to(dtype=torch.float16)
        return img

class VideoGame:

    def __init__(self):
        from controller import VirtualController
        self.video_state = VideoState()
        self.game_state = GameState()
        self.virtual_controller = VirtualController

    def act(self, action: tuple):
        self.virtual_controller.update(action)

    def observe(self):
        self.game_state.set_flag(FLAGS.RESET, False)
        velocity, collided = self.game_state.pop()
        video_state = self.video_state.pop()
        truncated = 0
        return (video_state, velocity, collided, truncated)

class Environment(Env):

    def __init__(self, env_config=None):
        self.video_game = VideoGame()
        self.action_space = Box(low=-1.0, high=1.0, shape=config.action_space_shape)
        self.observation_space = Tuple([
            Box(low=0, high=1, shape=config.observation_space_shape['image']),
            Box(low=-float('inf'), high=float('inf'), shape=config.observation_space_shape['velocity'])
        ])

    def step(self, action):
        self.video_game.act(action)
        video_state, velocity, collided, truncated = self.video_game.game_state.observe()
        reward = 0 if collided == 0 else -1
        terminal = collided != 0
        return (
            (video_state, velocity),
            reward,
            terminal,
            truncated,
            {}
        )

    def reset(self, *args, **kwargs):
        self.video_game.game_state.set_flag(FLAGS.RESET, True)
        return self.video_game.observe()[:2], {"env_state": "reset"}