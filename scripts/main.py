import argparse

args = argparse.ArgumentParser()
# TODO: use git for model version control
# i.e. have a main branch without model.py. when developing model, create new branch named 'xyz' with model.py. save & commit when ready to run algo.
# when `--train model_name run` is run, switch to the `model_name` branch and pull config from model.py. if `run` is omitted, use previous_run if it exists otherwise use `default`.
args.add_argument('--train', nargs='*', help='--train [model_name]\nLoads model_name if it is provided and exists in the checkpoint directory, otherwise creates it.\nIf no arguments are provided, loads the last run model if available, otherwise defaults to `ppo`.')
args.add_argument('--nosave', action='store_true', help='If set, does not save the or log run.')
args.add_argument('--flags', action='append', nargs='*', type=int, help='--flags\nPrints IPC flag values.\n--flags FLAG\nFlips flag.\n--flags FLAG VALUE\nSets flag.\nNOTE: Flipping a flag is ignored if that flag is set, and printing occurs after all operations.')
args.add_argument('--unlock', action='store_true', help='Switches whether the game is synchronized to the model.')

args = args.parse_args()

if args.flags is not None:
    from ipc import Flags
    F = Flags()
    commands = {}
    for command in args.flags:
        if len(command) == 1:
            commands[command[0]] = None
    for command in args.flags:
        if len(command) == 2:
            commands[command[0]] = command[1]
    for flag, value in commands.items():
        if value is None:
            F.set_flag(flag, not F.get_flag(flag))
        else:
            F.set_flag(flag, value)
    if [] in args.flags:
        Flags.debug()

if args.unlock:
    from ipc import Flags, GLOBAL_FLAGS
    GLOBAL_FLAGS.set_flag(Flags.UNSTUCK, not GLOBAL_FLAGS.get_flag(Flags.UNSTUCK))
    GLOBAL_FLAGS.set_flag(Flags.REQUEST_GAME_STATE, True)


if args.train is not None:
    import gc
    import os
    from pathlib import Path
    import config
    import torch
    from ray.tune.logger import UnifiedLogger
    from environment import Environment, VideoState
    from ipc import Flags, GLOBAL_FLAGS
    from util import PROJECT_DIR
    from ray_misc import LassoLearner, ZeroCrashRewardLearnerConnector
    from model import model_config

    MODEL_NAME = 'ppo'
    LAST_RUN = Path(PROJECT_DIR) / 'last_run'

    if LAST_RUN.exists():
        with open(LAST_RUN, 'r') as file:
            MODEL_NAME = file.read()

    if len(args.train) >= 1:
        MODEL_NAME = args.train[0]

    LOG_DIR = Path(PROJECT_DIR) / 'ray_results' / MODEL_NAME
    CHECKPOINT_DIR = Path(PROJECT_DIR) / 'checkpoints' / MODEL_NAME

    algorithm = (
        model_config
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
        .build_algo(logger_creator=None if args.nosave else lambda config: UnifiedLogger(config, logdir=LOG_DIR, loggers=None))
    )

    if CHECKPOINT_DIR.exists():
        algorithm.load_checkpoint(str(CHECKPOINT_DIR))
        # fix for https://github.com/ray-project/ray/issues/51560
        def betas_tensor_to_float(learner):
            param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
            param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)

    print("Waiting for script to load")
    GLOBAL_FLAGS.wait_until(Flags.BEGIN_TRAINING, True)
    print("Script loaded")
    VideoState.init_cuda_arrays()
    last_run_written = False
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        results = algorithm.train()
        if not args.nosave:
            algorithm.save_checkpoint(str(CHECKPOINT_DIR))
            if not last_run_written:
                with open(Path(PROJECT_DIR) / 'last_run', 'w') as file:
                    file.write(MODEL_NAME)
                last_run_written = True
