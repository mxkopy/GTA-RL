import torch
import config
import numpy as np
from struct import unpack
from ipc import Flags, GLOBAL_FLAGS, StructuredMemory, RequestLockedMemory
from cuda_ipc import CUDAArray
from gymnasium import Env
from gymnasium.spaces import Tuple, Box, Discrete
from math import prod

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

    NEAR = 0.15
    FAR = 10003.815

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
        dim_pads = (1 for _ in range(4 - len(img.shape)))
        img = img.reshape(*dim_pads, *img.shape)
        size = config.image_shape[1:]
        return torch.nn.functional.interpolate(img, size, mode='bilinear', antialias=True).squeeze()

    @staticmethod
    def pop_rgb():
        img = VideoState.CUDAArrays['RGB']
        img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        img = img / 255
        img = VideoState.rescale(img)
        img = img.squeeze()[:3, ...]
        return img

    @staticmethod
    def linearize_depth(array, cutoff=config.depth_cutoff):
        # VS = VideoState.VertexShaderConstantsMemory.data
        # n, f = VS.nearclip, VS.farclip
        # Pulled from https://github.com/umautobots/GTAVisionExport/issues/13#issuecomment-765115705
        Z = (-(VideoState.FAR*VideoState.NEAR)/(VideoState.NEAR-VideoState.FAR))/(array - (VideoState.NEAR/(VideoState.NEAR-VideoState.FAR)))
        if cutoff is not None:
            Z[Z >= cutoff] = cutoff
            Z = (Z - VideoState.NEAR) / (cutoff - VideoState.NEAR)
            return 1 - Z
        return Z

    @staticmethod
    def voxelize(x, depth=config.voxel_depth, min_val=NEAR/config.depth_cutoff, max_val=1.0):
        x = x.squeeze()
        if len(x.shape) > 2:
            xs = x.shape
            x = x.reshape(-1, xs[-2], xs[-1])
            vx = torch.stack([VideoState.voxelize(x[i, ...]) for i in range(prod(xs[:-2]))])
            vx = vx.reshape(*xs[:-2], -1, *xs[-2:])
            return vx
        z = torch.zeros_like(x, dtype=torch.bool).unsqueeze(0).repeat(depth, 1, 1)
        bounds = torch.linspace(min_val, max_val, depth+1, device=z.device) 
        lb = bounds[:-1].reshape(-1, 1, 1).repeat(1, z.shape[1], z.shape[2])
        ub = bounds[1:].reshape(-1, 1, 1).repeat(1, z.shape[1], z.shape[2])
        z[(lb <= x) & (x < ub)] = 1
        return z.unsqueeze(0).to(dtype=x.dtype)

    @staticmethod
    def pop_depth():
        buffer = VideoState.CUDAArrays['Depth']
        depth = VideoState.rescale(buffer.squeeze()[:, :, 3])
        return depth

    @staticmethod
    def pop_velocity():
        buffer = VideoState.CUDAArrays['Depth']
        return VideoState.rescale(buffer.squeeze().permute(2, 0, 1)[:, :, :3])

    @staticmethod
    def pop():
        depth = VideoState.pop_depth()
        depth = VideoState.linearize_depth(depth)
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
        # frame = self.video_state.pop().reshape(-1)
        # speed, collided = self.game_state.pop()
        return self.video_state.pop().reshape(-1), self.game_state.pop()

class Environment(Env):

    def __init__(self, env_config={'horizon': None}):
        self.video_game = VideoGame()
        self.action_space = Box(low=-1.0, high=1.0, shape=config.action_space_shape)
        self.observation_space = Box(low=0.0, high=1.0, shape=config.observation_space_shape)
        self.last_n_frames = []
        self.horizon = env_config['horizon']
        self.t = 0

    def calculate_reward(self, game_state):
        speed, collided = game_state
        horizon = 1 if self.horizon is None else self.horizon
        return -1000 if collided else np.log10(1 + np.abs(speed)) * np.sign(speed)

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



# Custom logger to record environment data 
from ray.rllib.callbacks.callbacks import RLlibCallback
class EpisodeStepCallback(RLlibCallback):

    def __init__(self):
        super().__init__()
        self.VSBMemory = StructuredMemory("VSConstants")
        vsb = self.VSBMemory.data.constant_buffers[2]
        vsb = np.frombuffer(vsb, dtype=np.float32).reshape(-1, 4)

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["camera_position"] = []
        episode.custom_data["camera_direction"] = []

    def on_episode_step(self, *, episode, **kwargs):
        vsb = np.frombuffer(self.VSBMemory.data.constant_buffers[2], dtype=np.float32)
        episode.custom_data["camera_position"].append(tuple(vsb[12:15]))
        episode.custom_data["camera_direction"].append(tuple(vsb[16:19]))

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        metrics_logger.log_value("camera_position", value=episode.custom_data["camera_position"], reduce="item")
        metrics_logger.log_value("camera_direction", value=episode.custom_data["camera_direction"], reduce="item")

