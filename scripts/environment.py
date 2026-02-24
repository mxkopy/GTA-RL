import torch
import random
import config
import torchvision
import math
import mmap
import cupy
import numpy as np
import structs
from struct import unpack
from ipc import Flags, StructuredMemory, RequestLockedMemory
from cuda_ipc import CUDAArray
from gymnasium import Env
from gymnasium.spaces import Tuple, Box, Discrete

class GameState:

    GameStateMemory = RequestLockedMemory('GameState', Flags.REQUEST_GAME_STATE)

    @staticmethod
    def pop():
        state = GameState.GameStateMemory.data
        return (
            (state.camera_direction.x, state.camera_direction.y, state.camera_direction.z),
            (state.velocity.x, state.velocity.y, state.velocity.z),
            state.collided
        )


class VideoState:

    VertexShaderConstantsMemory = StructuredMemory('VSConstantBuffers')
    
    CUDAArrays = {
        'Depth': None,
        'RGB': None
    }

    @staticmethod
    def init_cuda_arrays():
        CUDAArrays = VideoState.CUDAArrays
        for name in CUDAArrays:
            if CUDAArrays[name] is None:
                CUDAArrays[name] = torch.from_dlpack(CUDAArray(name).data)

    def rescale(img: torch.Tensor):
        img = img.squeeze()
        size = config.image_shape[1:]
        return torch.nn.functional.interpolate(img.unsqueeze(0), size, mode='bilinear', antialias=True).squeeze()

    def linearize_depth(array):
        VS = VideoState.VertexShaderConstantsMemory.data
        n, f = VS.nearclip, VS.farclip
        z = (f - n) / (array * (f + n))
        return z

    @staticmethod
    def pop_rgb():
        VideoState.init_cuda_arrays()
        img = VideoState.CUDAArrays['RGB']
        img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        img = img / 255
        img = VideoState.rescale(img)
        img = img.squeeze()[:3, ...]
        return img

    @staticmethod
    def pop_depth_and_velocity():
        VideoState.init_cuda_arrays()
        depthinfo = VideoState.CUDAArrays['Depth']
        # near, far = unpack('@2f', VideoState.nearclipfarclip.pop_nbl())
        velocity_and_depth = depthinfo.squeeze().permute(2, 0, 1)
        velocity_and_depth = VideoState.rescale(velocity_and_depth)
        velocity_and_depth = torch.nan_to_num(velocity_and_depth, posinf=0, neginf=0)
        # print(velocity_and_depth[:3, ...].max(), velocity_and_depth[:3, ...].min())
        return velocity_and_depth[:3, ...], velocity_and_depth[3, ...].unsqueeze(0)

    @staticmethod
    def pop():
        velocity, depth = VideoState.pop_depth_and_velocity()
        rgb = VideoState.pop_rgb()
        img = torch.cat((depth, rgb))
        return img.cpu()

class VideoGame:

    def __init__(self):
        from controller import VirtualController
        self.video_state = VideoState
        self.game_state = GameState
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
        observation = torch.cat((video_state.reshape(-1), torch.tensor(velocity, device=video_state.device)) )
        reward = VideoGame.reward(camera_direction, velocity, collided)
        terminal = collided == 0
        truncated = False
        return observation.reshape(-1), reward, terminal, truncated

class Environment(Env):

    def __init__(self, conf=None):
        self.device = 'cuda'
        self.video_game = VideoGame()
        self.action_space = Box(low=-1.0, high=1.0, shape=config.action_space_shape)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=config.observation_space_shape)

    def step(self, action):
        self.video_game.act(action)
        observation, reward, terminal, truncated = self.video_game.observe()
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