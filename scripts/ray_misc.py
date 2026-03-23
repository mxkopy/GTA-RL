
import numpy as np
from collections.abc import Iterable
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.connectors.connector_v2 import ConnectorV2

# Mixed precision stuff
HALF = np.float16
def astype(x, T):
    if hasattr(x, '__array__'):    
        return np.astype(x, T)
    else:
        if isinstance(x, dict):
            for k in x:
                x[k] = astype(x[k], T)
        elif isinstance(x, Iterable):
            return x.__class__((astype(y, T) for y in x))
    return x

# Blatant https://github.com/ray-project/ray/blob/master/rllib/examples/gpus/mixed_precision_training_float16_inference.py
def halfp_algorithm(algorithm: Algorithm, **kwargs):
    algorithm.env_runner_group.foreach_env_runner(
        lambda env_runner: env_runner.module.half()
    )
    if algorithm.eval_env_runner_group:
        algorithm.eval_env_runner_group.foreach_env_runner(
            lambda env_runner: env_runner.module.half()
        )

class Float16Connector(ConnectorV2):
    def recompute_output_observation_space(self, input_observation_space, input_action_space):
        from gymnasium.spaces import Box
        if hasattr(input_observation_space, 'spaces'):
            if isinstance(input_observation_space.spaces, dict):
                for space in input_observation_space.spaces.values():
                    self.recompute_output_observation_space(space, input_action_space)
            else:
                for space in input_observation_space.spaces:
                    self.recompute_output_observation_space(space, input_action_space)
        elif isinstance(input_observation_space, Box):
            input_observation_space.high, input_observation_space.bounded_above = input_observation_space._cast_high(input_observation_space.high, float(np.finfo(HALF).max))
            input_observation_space.low, input_observation_space.bounded_below = input_observation_space._cast_low(input_observation_space.low, float(np.finfo(HALF).min))
            input_observation_space.dtype = np.dtype(HALF)
        return input_observation_space

    def __call__(self, *, rl_module, batch, episodes, explore = None, shared_data = None, metrics = None, **kwargs):
        for sa_episode in self.single_agent_episode_iterator(episodes):
            obs = sa_episode.get_observations(-1)
            half_obs = astype(obs, HALF)
            self.add_batch_item(batch, column="obs", item_to_add=half_obs, single_agent_episode=sa_episode)
        return batch

# Normalizes episode rewards
class NormalizeRewards(ConnectorV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        for sa_episode in self.single_agent_episode_iterator(episodes=episodes, agents_that_stepped_only=False):
            rewards = sa_episode.get_rewards()
            for i, r in enumerate(rewards):
                sa_episode.set_rewards(
                    new_data=(r - np.mean(rewards))/(np.std(rewards) + 1e-10),
                    at_indices=i
                )
        return batch
