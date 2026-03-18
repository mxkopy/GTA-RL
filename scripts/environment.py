import torch
import config
import numpy as np
from struct import unpack
from ipc import Flags, GLOBAL_FLAGS, StructuredMemory, RequestLockedMemory
from cuda_ipc import CUDAArray
from gymnasium import Env
from gymnasium.spaces import Tuple, Box, Discrete    

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
    def linearize_depth(array):
        # VS = VideoState.VertexShaderConstantsMemory.data
        # n, f = VS.nearclip, VS.farclip
        NEAR = 0.15
        FAR = 10003.815
        n, f = NEAR, FAR
        # Pulled from https://github.com/umautobots/GTAVisionExport/issues/13#issuecomment-765115705
        Z = (-(FAR*NEAR)/(NEAR-FAR))/(array - (NEAR/(NEAR-FAR)))
        return Z

    @staticmethod
    def cutoff_depth(array, cutoff=1000):
        array[array >= cutoff] = 0
        return array

    @staticmethod
    def pop_rgb():
        img = VideoState.CUDAArrays['RGB']
        img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        img = img / 255
        img = VideoState.rescale(img)
        img = img.squeeze()[:3, ...]
        return img

    # Velocity map scaled by depth (3 dims)
    # Distance map with cutoff (1 dim) -> 4 dim input
    @staticmethod
    def pop_velocity_and_depth(cutoff=100.0):
        buffer = VideoState.CUDAArrays['Depth']
        velocity_and_depth = buffer.squeeze().permute(2, 0, 1)
        velocity_and_depth = VideoState.rescale(velocity_and_depth)
        velocity = velocity_and_depth[:3, ...]
        depth = velocity_and_depth[3, ...].unsqueeze(0)
        return depth
        # distances = VideoState.linearize_depth(depth)
        # distances[distances >= cutoff] = cutoff
        # depth[distances >= cutoff] = 0
        # return velocity / 1e20, depth, distances

    @staticmethod
    def pop():
        depth = VideoState.pop_velocity_and_depth()
        return depth.cpu()
        # velocity, depth, distances = VideoState.pop_velocity_and_depth()
        # rgb = VideoState.pop_rgb()
        # return torch.cat((depth, distances, velocity)).cpu()

class VideoGame:

    def __init__(self):
        from controller import VirtualController
        self.video_state = VideoState
        self.game_state = GameState
        self.virtual_controller = VirtualController

    def act(self, action: tuple):
        self.virtual_controller.update(action)

    def observe(self):
        return self.video_state.pop().reshape(-1), self.game_state.pop()

class Environment(Env):

    def __init__(self, env_config={'horizon': None}):
        self.video_game = VideoGame()
        self.action_space = Box(low=-1.0, high=1.0, shape=config.action_space_shape)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=config.observation_space_shape)
        self.last_n_frames = []
        self.horizon = env_config['horizon']
        self.t = 0

    def calculate_reward(self, game_state):
        speed, collided = game_state
        horizon = 1 if self.horizon is None else self.horizon
        return -1 if collided else max(0, speed) / horizon

    def truncate(self) -> bool:
        if self.horizon is not None:
            self.t += 1
            return self.t >= self.horizon
        return False

    def stack_frame(self, observation):
        if config.n_frames <= 1:
            return observation.reshape(-1)
        self.last_n_frames = [observation] + self.last_n_frames[:-1]
        return torch.stack(self.last_n_frames, dim=0).reshape(-1)

    def step(self, action):
        self.video_game.act(action)
        video_state, game_state = self.video_game.observe()
        observation = self.stack_frame(video_state)
        reward = self.calculate_reward(game_state)
        terminal = game_state[1]
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
        self.t = 0
        self.video_game.game_state.reset()
        frame = self.video_game.observe()[0]
        self.last_n_frames = [torch.zeros_like(frame) for _ in range(config.n_frames)]
        observation = self.stack_frame(frame)
        return observation, {"env_state" : "reset"}