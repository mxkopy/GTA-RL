import torch
import random
import config
import torchvision
import math
import mmap
import cupy
import numpy as np
from struct import unpack
from ipc import Flags, FLAGS, Channel, Message, Queue, MessageMapped, Mutex
from gymnasium import Env
from gymnasium.spaces import Tuple, Box, Discrete
import msgs_pb2

class GameState(Queue, metaclass=MessageMapped):

    LOCK = FLAGS.REQUEST_GAME_STATE
    TAGNAME = 'game_state.ipc'
    MSG_TYPE = msgs_pb2.GameState

    def __init__(self):
        super().__init__(GameState.N_BYTES, GameState.TAGNAME)

    def pop(self):
        game_state = super().pop()
        return (
            (game_state.camera_direction.x, game_state.camera_direction.y, game_state.camera_direction.z),
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
        return torch.nn.functional.interpolate(img, config.observation_space_shape['image'][1:], mode='bilinear', antialias=True)

    def pop_rgb():
        VideoState.init_cuda_array("RGB")
        img = VideoState.tensors["RGB"]
        img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        img = img / 255
        img = VideoState.rescale(img)
        img = img.squeeze()[:3, ...]
        return img

    def linearize_depth(array):
        near, far = unpack('@2f', VideoState.nearclipfarclip.pop_nbl())
        y = (torch.pow(far/near,array)-1) * (near / far)
        depth = ((y * near) / far)
        return depth

    def pop_depth():
        VideoState.init_cuda_array("DepthBuffer")
        depth = VideoState.tensors["DepthBuffer"].squeeze()
        depth = VideoState.linearize_depth(depth)
        return VideoState.rescale(depth.unsqueeze(0).unsqueeze(0)).squeeze()

    def pop() -> torch.Tensor:
        depth = VideoState.pop_depth().unsqueeze(0)
        rgb = VideoState.pop_rgb()
        img = torch.cat((depth, rgb))
        return img.cpu()

class VideoGame:

    def __init__(self):
        from controller import VirtualController
        self.video_state = VideoState
        self.game_state = GameState()
        self.virtual_controller = VirtualController

    def act(self, action: tuple):
        self.virtual_controller.update(action)

    def reward(camera_direction, velocity, collided):
        if collided:
            return -10
        else:
            return np.dot(np.array(camera_direction), np.array(velocity))

    def observe(self):
        camera_direction, velocity, collided = self.game_state.pop()
        video_state = self.video_state.pop()
        reward = VideoGame.reward(camera_direction, velocity, collided)
        terminal = collided == 0
        truncated = False
        return (video_state, velocity), reward, terminal, truncated



class Environment(Env):

    def __init__(self, conf=None):
        self.device = 'cuda'
        self.video_game = VideoGame()
        self.action_space = Box(low=-1.0, high=1.0, shape=config.action_space_shape)
        self.observation_space = Tuple([
            Box(low=0, high=1, shape=config.observation_space_shape['image']),
            Box(low=-float('inf'), high=float('inf'), shape=config.observation_space_shape['velocity'])
        ])
        self.mutex = Mutex()

    def step(self, action):
        self.mutex.acquire()
        self.video_game.act(action)
        observation, reward, terminal, truncated = self.video_game.observe()
        self.mutex.release()
        print(f"{action[0]: >10.5f} {action[1]: >10.5f} {action[2]: >10.5f} | {str(reward)[0:5]}")
        return (
            observation,
            reward,
            terminal,
            truncated,
            {}
        )

    def reset(self, *args, **kwargs):
        obs = self.video_game.observe()[0], {"env_state": "reset"}
        return obs