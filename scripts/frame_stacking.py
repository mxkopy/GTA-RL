import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.utils.annotations import override
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns

class FrameStacking(ConnectorV2):

    @override(ConnectorV2)
    def recompute_output_observation_space(
        self,
        input_observation_space,
        input_action_space,
    ):
        assert (
            isinstance(input_observation_space, gym.spaces.Box)
            and len(input_observation_space.shape) == 1
        )
        return gym.spaces.Box(
            low=np.repeat(input_observation_space.low, repeats=self.n_frames),
            high=np.repeat(input_observation_space.high, repeats=self.n_frames),
            shape=(input_observation_space.shape[0] * self.n_frames,),
            dtype=input_observation_space.dtype,
        )

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        num_frames: int = 1,
        as_learner_connector: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_observation_space=input_observation_space, 
            input_action_space=input_action_space, 
            **kwargs
        )
        self.n_frames = num_frames
        self.as_learner_connector = as_learner_connector

    @override(ConnectorV2)
    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        for sa_episode in self.single_agent_episode_iterator(episodes):    
            last_n_obs = sa_episode.get_observations(
                indices=slice(-self.n_frames, None),
                fill=0.0,
            )
            new_obs = np.concatenate(last_n_obs, axis=0)
            self.add_batch_item(
                batch=batch,
                column=Columns.OBS,
                item_to_add=new_obs,
                single_agent_episode=sa_episode,
            )
        return batch