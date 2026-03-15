import torch
import config
import numpy as np
from struct import unpack
from ipc import Flags, GLOBAL_FLAGS, StructuredMemory, RequestLockedMemory
from cuda_ipc import CUDAArray
from gymnasium import Env
from gymnasium.spaces import Tuple, Box, Discrete
from math import sqrt        

class GameState:

    GameStateMemory = RequestLockedMemory('GameState', Flags.REQUEST_GAME_STATE)

    @staticmethod
    def pop():
        state = GameState.GameStateMemory.data
        return state.reward, state.collided

    @staticmethod
    def reset():
        GameState.GameStateMemory.flags.set_flag(Flags.REQUEST_GAME_STATE, True)
        GameState.GameStateMemory.flags.set_flag(Flags.RESET, True)
        GameState.GameStateMemory.flags.wait_until(Flags.RESET, False)

class VideoState:

    VertexShaderConstantsMemory = StructuredMemory('VSConstants')
    
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
    
    @staticmethod
    def rescale(img: torch.Tensor):
        img = img.squeeze()
        size = config.image_shape[1:]
        return torch.nn.functional.interpolate(img.unsqueeze(0), size, mode='bilinear', antialias=True).squeeze()

    @staticmethod
    def linearize_depth(array, cutoff=100.0):
        # VS = VideoState.VertexShaderConstantsMemory.data
        # n, f = VS.nearclip, VS.farclip
        NEAR = 0.15
        FAR = 10003.815
        n, f = NEAR, FAR
        Z = (-(f*n)/(n-f))/(array - (n/(n-f)))
        Z[Z >= cutoff] = cutoff
        return Z

    @staticmethod
    def pop_rgb():
        img = VideoState.CUDAArrays['RGB']
        img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        img = img / 255
        img = VideoState.rescale(img)
        img = img.squeeze()[:3, ...]
        return img

    @staticmethod
    def pop_velocity_and_depth():
        depthinfo = VideoState.CUDAArrays['Depth']
        # near, far = unpack('@2f', VideoState.nearclipfarclip.pop_nbl())
        velocity_and_depth = depthinfo.squeeze().permute(2, 0, 1)
        velocity_and_depth = VideoState.rescale(velocity_and_depth)
        velocity_and_depth = torch.nan_to_num(velocity_and_depth, posinf=0, neginf=0)
        # print(velocity_and_depth[:3, ...].max(), velocity_and_depth[:3, ...].min())
        return velocity_and_depth[:3, ...], velocity_and_depth[3, ...].unsqueeze(0)

    @staticmethod
    def pop():
        velocity, depth = VideoState.pop_velocity_and_depth()
        rgb = VideoState.pop_rgb()
        depth = VideoState.linearize_depth(depth)
        # img = torch.cat((depth, rgb))
        return depth.cpu()

class VideoGame:

    def __init__(self):
        from controller import VirtualController
        self.video_state = VideoState
        self.game_state = GameState
        self.virtual_controller = VirtualController

    def act(self, action: tuple):
        self.virtual_controller.update(action)

    def observe(self):
        reward, collided = self.game_state.pop()
        video_state = self.video_state.pop()
        observation = video_state.reshape(-1)
        reward = (-10 * collided) + (sqrt(abs(reward)) - 0.1) * (1 - collided) * (-1 if reward < 0 else 1)
        is_terminal = collided
        return observation, reward, is_terminal

class Environment(Env):

    def __init__(self, conf=None):
        self.video_game = VideoGame()
        self.action_space = Box(low=-1.0, high=1.0, shape=config.action_space_shape)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=config.observation_space_shape)
        self.last_n_frames = []
        self.horizon = float('inf')
        if conf is not None and 'horizon' in conf:
            self.horizon = conf['horizon']
            self.t = 0

    def truncate(self) -> bool:
        if hasattr(self, 't'):
            self.t += 1
            return self.t >= self.horizon
        return False

    def stack_observation(self, observation):
        if config.n_frames <= 1:
            return observation.reshape(-1)
        self.last_n_frames = [observation] + self.last_n_frames[:-1]
        return torch.stack(self.last_n_frames, dim=0).reshape(-1)

    def step(self, action):
        self.video_game.act(action)
        observation, reward, terminal = self.video_game.observe()
        observation = self.stack_observation(observation)
        truncated = self.truncate()
        # print(f"{action[0]: >10.5f} {action[1]: >10.5f} | {str(reward)[0:5]}")
        return (
            observation,
            reward,
            terminal,
            truncated,
            {}
        )

    def reset(self, *args, **kwargs):
        if hasattr(self, 't'):
            self.t = 0
        self.video_game.game_state.reset()
        obs = self.video_game.observe()[0]
        self.last_n_frames = [torch.zeros_like(obs) for _ in range(config.n_frames)]
        obs = self.stack_observation(obs)
        return obs, {"env_state" : "reset"}