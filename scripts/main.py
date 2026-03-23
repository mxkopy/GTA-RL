import argparse

args = argparse.ArgumentParser()
# TODO: use git for model version control
# i.e. have a main branch without model.py. when developing model, create new branch named 'xyz' with model.py. save & commit when ready to run algo.
# <changing model.py>
#
# > python main.py --train model_name
#
# if model_name is not an existing branch:
# > git switch -c model_name
# > git add scripts/main.py 
# > git commit --allow-empty-message
#
# if model_name is an existing branch:
# > git stash push
# > git switch model_name
#
# > git merge main         -> should auto merge w/ ort
#
# onexit: 
# if stash was used:
# > git stash pop 
args.add_argument('--train', nargs='*', help='--train [model_name]\nLoads model_name if it provided and exists in the checkpoint directory, otherwise creates it.\nIf no arguments are provided, loads the last run model if available, otherwise defaults to `ppo`.')
args.add_argument('--view', action='store_true', help='Shows the depth buffer and RGB view.')
args.add_argument('--nosave', action='store_true', help='If set, does not save the or log run.')
args.add_argument('--flags', action='append', nargs='*', type=int, help='--flags\nPrints IPC flag values.\n--flags FLAG\nFlips flag.\n--flags FLAG VALUE\nSets flag.\nNOTE: Flipping a flag is ignored if that flag is set, and printing occurs after all operations.')
args.add_argument('--unlock', action='store_true', help='Switches whether the game is synchronized to the model or free-running.')

args = args.parse_args()

def cmd_flags():
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

def cmd_unlock():
    from ipc import Flags, GLOBAL_FLAGS
    GLOBAL_FLAGS.set_flag(Flags.UNSTUCK, not GLOBAL_FLAGS.get_flag(Flags.UNSTUCK))
    GLOBAL_FLAGS.set_flag(Flags.REQUEST_GAME_STATE, True)

def cmd_train():
    from util import PROJECT_DIR
    import gc
    from pathlib import Path
    import config
    import torch
    from ray.tune.logger import UnifiedLogger
    from environment import Environment, VideoState
    from ipc import Flags, GLOBAL_FLAGS
    from model import model_config
    from ray_misc import NormalizeRewards

    MODEL_NAME = 'default'
    LAST_RUN = Path(PROJECT_DIR) / 'last_run'

    if LAST_RUN.exists():
        with open(LAST_RUN, 'r') as file:
            MODEL_NAME = file.read()

    if len(args.train) >= 1:
        MODEL_NAME = args.train[0]

    LOG_DIR = Path(PROJECT_DIR) / 'ray_results' / MODEL_NAME
    CHECKPOINT_DIR = Path(PROJECT_DIR) / 'checkpoints' / MODEL_NAME

    def to_list(x):
        if x is None:
            return []
        if not isinstance(x, list):
            if isinstance(x, tuple):
                return list(x)
            else:
                return [x]
        return x

    model_learner_connector = (lambda *args: []) if model_config._learner_connector is None else model_config._learner_connector

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
            learner_connector=lambda *args: to_list(model_learner_connector(*args)) + [NormalizeRewards()],
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

def cmd_view():
    import config
    import cv2
    import torch
    from environment import VideoState

    VideoState.init_cuda_arrays()

    print("Showing")
    while True:
        keypress = cv2.waitKey(1)
        rgb = VideoState.pop_rgb()
        depth = VideoState.pop_depth()
        depth = VideoState.linearize_depth(depth)
        voxels = VideoState.voxelize(depth).squeeze() 
        voxels = voxels * torch.arange(config.voxel_depth, device=voxels.device).reshape(-1, 1, 1)
        voxels = voxels.sum(dim=0)
        cv2.imshow("Voxels", voxels.cpu().numpy() / config.voxel_depth )
        cv2.imshow("Depth", depth.squeeze().cpu().numpy())
        cv2.imshow("RGB", rgb.permute(1, 2, 0).squeeze().cpu().numpy())

if __name__ == '__main__':
    import multiprocessing
    import signal

    if args.flags:
        cmd_flags()

    if args.unlock:
        cmd_unlock()

    if args.train:
        cmd_train()

    if args.view:
        cmd_view()
