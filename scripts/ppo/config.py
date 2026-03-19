import config
from environment import Environment
from model import PPODriver
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger

config = (
    PPOConfig()
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=PPODriver,
            observation_space=Environment().observation_space,
            action_space=Environment().action_space,
            model_config={
                'max_seq_len': config.minibatch_size,
            }
        )
    )
    .training(
        use_gae=True,
        use_critic=True,
        use_kl_loss=False,
        lr=config.learning_rate,
        gamma=config.gamma,
        train_batch_size=config.train_batch_size,
        minibatch_size=config.minibatch_size,
        num_epochs=config.num_epochs,
        lambda_=config.gae_lambda,
        clip_param=config.clip_param,
        entropy_coeff=config.entropy_coeff,
        vf_loss_coeff=config.vf_loss_coeff
    )
)