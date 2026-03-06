import sys

if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'train':
        import gc
        import torch
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.algorithms.ppo import PPOConfig
        from environment import Environment, VideoState
        from ipc import Flags, PROJECT_DIR
        from model import Model
        from pathlib import Path

        TRAIN_BATCH_SIZE = 128
        MINIBATCH_SIZE = 32
        MODEL_PATH = Path(PROJECT_DIR) / 'driver.ckpt'

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
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=0.5,
            )
            .environment(
                env=Environment,
                # env_config={'horizon': 256}
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=Model,
                    observation_space=Environment().observation_space,
                    action_space=Environment().action_space,
                    model_config={
                        'max_seq_len': MINIBATCH_SIZE,
                    }
                )
            )
            .training(
                lr=1e-3,
                train_batch_size=TRAIN_BATCH_SIZE,
                minibatch_size=MINIBATCH_SIZE,
                num_epochs=3,
                use_kl_loss=False,
                clip_param=0.1,
                entropy_coeff=0.001,
                vf_loss_coeff=1,
                kl_target=0.003,
            )
            .build_algo()
        )
        if MODEL_PATH.exists():
            algorithm.load_checkpoint(str(MODEL_PATH))
            # fix for https://github.com/ray-project/ray/issues/51560
            def betas_tensor_to_float(learner):
                param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
                param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
            algorithm.learner_group.foreach_learner(betas_tensor_to_float)
        flags = Flags()
        print("Waiting for script to load")
        flags.wait_until(Flags.BEGIN_TRAINING, True)
        print("Script loaded")
        VideoState.init_cuda_arrays()
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            results = algorithm.train()
            algorithm.save_checkpoint(str(MODEL_PATH))

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

    if sys.argv[1] == 'unstuck':
        from ipc import Flags
        flags = Flags()
        flags.set_flag(Flags.UNSTUCK, not flags.get_flag(Flags.UNSTUCK))
        flags.set_flag(Flags.REQUEST_GAME_STATE, True)
