# import argparse

# args = argparse.ArgumentParser()

# args.add_argument('--train', nargs='?', const='ppo', help='--train [ppo|sac]')
# args.add_argument('--continue', nargs='?', help='--continue [model_name]\nContinues training the specified model. If no model is specified, continues training the most recent model. Does nothing if --train is not specified')
# args.add_argument('--debug', action='store_true', help='Prints IPC flags\n')
# args.add_argument('--flag', action='append', nargs=2, help='--flag FLAG VALUE\nSets IPC flag values')

# args = args.parse_args()
# print(args)
# exit()


import sys
if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'train':
        import gc
        import os
        from datetime import datetime
        from pathlib import Path
        import config
        import torch
        from ray.tune.logger import UnifiedLogger
        from environment import Environment, VideoState
        from ipc import Flags
        from util import PROJECT_DIR, LassoLearner, ZeroCrashRewardLearnerConnector
        from ppo.config import config as ppo_config

        TIME_FMT = '%Y-%m-%d_%H-%M-%S'
        ID = datetime.today().strftime(TIME_FMT)

        LOG_DIR = Path(PROJECT_DIR) / 'ray_results'
        CHECKPOINT_DIR = Path(PROJECT_DIR) / 'checkpoints'

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        if '--continue' in sys.argv:
            checkpoints = os.listdir(CHECKPOINT_DIR)
            if len(checkpoints) > 0:
                checkpoints.sort(key=lambda x: datetime.strptime(x, TIME_FMT))
                ID = checkpoints[-1]
        
        def logger_creator(config):
            log = str(LOG_DIR / ID)
            os.makedirs(log, exist_ok=True)
            return UnifiedLogger(config, logdir=log, loggers=None)

        algorithm = (
            ppo_config
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
                learner_class=LassoLearner,
                learner_config_dict={
                    "lasso_coeff": config.lasso_coeff
                },
                learner_connector=lambda observation_space, action_space: ZeroCrashRewardLearnerConnector(),
                num_learners=0,
                num_gpus_per_learner=0.5
            )
            .environment(
                env=Environment,
                env_config={
                    'horizon': config.horizon
                }
            )
            .build_algo(logger_creator=logger_creator)
        )

        CKPT = CHECKPOINT_DIR / ID
        if CKPT.exists():
            algorithm.load_checkpoint(str(CKPT))
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
            algorithm.save_checkpoint(str(CKPT))

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
