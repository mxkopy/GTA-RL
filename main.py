import threading
import io
import multiprocessing
import config
import time
import random
import sys
from collections.abc import Callable, Buffer
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
# from model import DriverActorModel, DriverCriticModel

if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'train':
        # from model import DriverActorModel, DriverCriticModel
        from environment import Environment
        from ipc import Flags, FLAGS
        from ray.rllib.algorithms.ppo import PPOConfig
        from model import Model
        algorithm = (
            PPOConfig()
            .framework("torch")
            .resources(num_gpus=1)
            .environment(env=Environment)
            .rl_module(rl_module_spec=RLModuleSpec(module_class=Model))
            .training(lr=1e-5)
            .build_algo()
        )
        # actor = DriverActorModel().to(device=config.device)
        # critic = DriverCriticModel().to(device=config.device)
        print("Waiting for script to load")
        # Flags().wait_until(FLAGS.REQUEST_ACTION, True)
        print("Script loaded")
        while True:
            algorithm.train()

    if sys.argv[1] == 'debug':
        from ipc import debug_flags, Flags
        if len(sys.argv) == 2:
            debug_flags()
        else: 
            flags = Flags()
            if sys.argv[3] == '0':
                flags.set_flag(int(sys.argv[2]), False)
            elif sys.argv[3] == '1':
                flags.set_flag(int(sys.argv[2]), True)
            
