import config
import sys
from ray.rllib.connectors.common.frame_stacking import FrameStacking


if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'train':
        import gc
        import torch
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.algorithms.ppo import PPOConfig
        from environment import Environment, VideoState
        from ipc import Flags
        from model import Model
        # from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
        # from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
        # from ray.rllib.connectors.common.frame_stacking import FrameStacking

        algorithm = (
            PPOConfig()
            .framework(
                "torch",
                torch_compile_learner=True,
                torch_compile_worker=True,
                torch_skip_nan_gradients=True
            )
            .resources(
                num_gpus=1,
            )
            .env_runners(
                num_env_runners=0,
                num_gpus_per_env_runner=0.5,
                # env_to_module_connector=lambda env, spaces, device: FrameStacking(n_frames=config.n_frames)
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=0.5,
            )
            .environment(
                env=Environment
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(module_class=Model)
            )
            .training(
                lr=1e-5,
                train_batch_size=64,
                minibatch_size=4,
                num_epochs=3,
                use_kl_loss=False,
                clip_param=0.1,
                entropy_coeff=0.001,
                vf_loss_coeff=1,
                kl_target=0.003,
                # grad_clip=None,
                # learner_connector=lambda obs_space, act_space: FrameStacking(num_frames=config.n_frames, as_learner_connector=True),
            )
            .build_algo()
        )
        print("Waiting for script to load")
        Flags().wait_until(Flags.BEGIN_TRAINING, True)
        print("Script loaded")
        VideoState.init_cuda_arrays()
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            algorithm.train()

    if sys.argv[1] == 'debug':
        from ipc import Flags
        commands = sys.argv[2:]
        if len(commands) > 0:
            flag = int(commands[0])
            if len(commands) > 1:
                value = int(commands[1])
                Flags.debug(flag=flag, value=value)
            else:
                Flags.debug(flag=flag)
        else:
            Flags.debug()

    if sys.argv[1] == 'fix':
        from ipc import Flags
        Flags.debug(Flags.BEGIN_TRAINING, False)
        Flags.debug(Flags.REQUEST_GAME_STATE, True)

    if sys.argv[1] == 'reset':
        from ipc import Flags
        Flags.debug(Flags.BEGIN_TRAINING, True)
        Flags.debug(Flags.REQUEST_GAME_STATE, False)
