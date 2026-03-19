import config
from environment import Environment
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.sac import SACConfig

config = (
    SACConfig()
    .rl_module(
        model_config=DefaultModelConfig()
    )
    .training(
        lr=1e-5,
        gamma=0.995,
        train_batch_size=config.train_batch_size,
        minibatch_size=config.minibatch_size,
        num_epochs=config.num_epochs,
        tau=0.05,
        # replay_buffer_config={
        #     '_enable_replay_buffer_api': True,
        #     'capacity': 256,
        #     'replay_batch_size': config.train_batch_size,
        #     'prioritized_replay_alpha': 0.6,
        #     'prioritized_replay_beta': 0.4,
        #     'prioritized_replay_eps': 1e-6,
        #     'replay_sequence_length': 1
        # },
        # optimization_config={
        #     'actor_learning_rate': 1e-5,
        #     'critic_learning_rate': 1e-5,
        #     'entropy_learning_rate': 1e-5
        # },
        actor_lr=3e-5,
        critic_lr=3e-4,
        alpha_lr=3e-4
        target_network_update_freq=256,
        num_steps_sampled_before_learning_starts=config.train_batch_size,
        twin_q=False,
        clip_actions=False,
        grad_clip=None,
        n_step=1,
        target_entropy='auto',
        initial_alpha=None,
        store_buffer_in_checkpoints=True
    )
)