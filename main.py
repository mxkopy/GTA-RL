import sys
from collections.abc import Iterable

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




if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'train':
        import gc
        import torch
        import numpy as np
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.connectors.connector_v2 import ConnectorV2
        from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.rllib.algorithms.ppo import PPOConfig
        from environment import Environment
        from ipc import Flags, FLAGS
        from model import Model
        # from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
        # from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
        # from ray.rllib.connectors.common.frame_stacking import FrameStacking

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

        algorithm = (
            PPOConfig()
            .framework(
                "torch",
                torch_compile_learner=True,
                torch_compile_worker=True,
                torch_skip_nan_gradients=True
            )
            .callbacks(
                # on_algorithm_init=halfp_algorithm
            )
            .resources(
                num_gpus=1,
            )
            .env_runners(
                num_env_runners=0,
                num_gpus_per_env_runner=0.5,
                # env_to_module_connector=lambda env, spaces, device: Float16Connector()
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=0.5,
                # learner_class=PPOTorchMixedPrecisionLearner
            )
            .environment(
                env=Environment
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(module_class=Model)
            )
            .training(
                lr=1e-5,
                train_batch_size=128,
                minibatch_size=16,
                num_epochs=3,
                use_kl_loss=True,
                clip_param=0.1,
                entropy_coeff=0.001,
                vf_loss_coeff=1,
                kl_target=0.003
                # grad_clip=None
            )
            # .experimental(
                # _torch_grad_scaler_class=torch.amp.GradScaler
            # )
            .build_algo()
        )
        print("Waiting for script to load")
        Flags().wait_until(FLAGS.BEGIN_TRAINING, True)
        print("Script loaded")
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            algorithm.train()

    if sys.argv[1] == 'debug':
        import time
        from ipc import debug_flags, Flags, FLAGS
        commands = sys.argv[2:]
        if len(commands) == 0:
            debug_flags()
        else: 
            flags = Flags()
            if commands[0] == 'fix':
                flags.set_flag(FLAGS.BEGIN_TRAINING, False)
                flags.set_flag(FLAGS.REQUEST_GAME_STATE, True)
            elif commands[0] == 'reset':
                flags.set_flag(FLAGS.BEGIN_TRAINING, True)
                flags.set_flag(FLAGS.REQUEST_GAME_STATE, False)
            else:
                flag = int(commands[0])
                value = not flags.get_flag(flag) if len(commands) <= 1 else int(commands[1])
                flags.set_flag(flag, value)