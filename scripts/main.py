import sys

if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'train':
        import gc
        import os
        import datetime
        from pathlib import Path
        import tempfile
        import torch
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.logger import UnifiedLogger
        from environment import Environment, VideoState
        from ipc import Flags, PROJECT_DIR
        from model import Model, Learner

        TRAIN_BATCH_SIZE = 128
        MINIBATCH_SIZE = 32
        MODEL_PATH = Path(PROJECT_DIR) / 'driver.ckpt'
        LOG_DIR = Path(PROJECT_DIR) / 'ray_results'
        LOG_NAME = f'driver_{datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        
        def logger_creator(config):
            if not Path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            logdir = tempfile.mkdtemp(prefix=LOG_NAME, dir=LOG_DIR)
            return UnifiedLogger(config, logdir=logdir, loggers=None)

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
                learner_class=Learner,
                learner_config_dict={
                    "regularizer_coeff": 1e-5
                },
                num_learners=0,
                num_gpus_per_learner=0.5
            )
            .environment(
                env=Environment,
                env_config={
                    'horizon': 256
                }
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
                lr=1e-5,
                train_batch_size=TRAIN_BATCH_SIZE,
                minibatch_size=MINIBATCH_SIZE,
                num_epochs=3,
                use_gae=True,
                use_critic=True,
                lambda_=0.9,
                clip_param=0.2,
                entropy_coeff=0.0001,
                vf_loss_coeff=1,
                use_kl_loss=False
            )
            .build(
                logger_creator=logger_creator
            )
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
