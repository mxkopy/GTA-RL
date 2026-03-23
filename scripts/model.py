import config
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

model_config = (
    PPOConfig()
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
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[64, 64, 64, 64],
            fcnet_activation='relu',
            use_lstm=True,
            max_seq_len=config.minibatch_size
        )
    )
)
