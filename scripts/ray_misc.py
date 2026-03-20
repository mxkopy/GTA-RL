
import torch
import numpy as np
from collections.abc import Iterable
from typing import Dict, Any, Optional
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.typing import ModuleID, TensorType
from ray.rllib.utils.annotations import override


# Custom learner adding L1 weight regularization
class LassoLearner(PPOTorchLearner):

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:

        base_total_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        # Compute the mean of all the RLModule's weights' absolute values.
        parameters = self.get_parameters(self.module[module_id])
        mean_abs_weight = torch.mean(torch.cat([p.reshape(-1).abs() for p in parameters]))

        self.metrics.log_value(
            key=(module_id, "lasso_coeff"),
            value=mean_abs_weight,
            window=1,
        )

        total_loss = (
            base_total_loss
            + config.learner_config_dict["lasso_coeff"] * mean_abs_weight
        )

        return total_loss
    
# Custom learner-connector to zero out rewards in episodes where the agent crashes
class ZeroCrashRewardLearnerConnector(ConnectorV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        for sa_episode in self.single_agent_episode_iterator(
            episodes=episodes, agents_that_stepped_only=False
        ):
            if sa_episode.is_terminated:
                rewards = sa_episode.get_rewards()
                for i, _ in enumerate(rewards):
                    sa_episode.set_rewards(
                        new_data=0,
                        at_indices=i
                    )
        return batch


# Mixed precision stuff
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

HALF = np.float16
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

class PPOTorchMixedPrecisionLearner(PPOTorchLearner):
    def _update(self, *args, **kwargs):
        with torch.amp.autocast("cuda", dtype=getattr(torch, HALF.__name__)):
            results = super()._update(*args, **kwargs)
        return results