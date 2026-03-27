import config
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

learning_rate=1e-5
gamma=0.995 
num_epochs=1
train_batch_size=128
minibatch_size=32
horizon=train_batch_size * 8
gae_lambda=0.95
clip_param=0.2
entropy_coeff=1e-5
vf_loss_coeff=1
lasso_coeff=1e-5

model_config = (
    PPOConfig()
    .training(
        use_gae=True,
        use_critic=True,
        use_kl_loss=False,
        lr=learning_rate,
        gamma=gamma,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
        lambda_=gae_lambda,
        clip_param=clip_param,
        entropy_coeff=entropy_coeff,
        vf_loss_coeff=vf_loss_coeff
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[64, 64, 64, 64],
            fcnet_activation='relu',
            use_lstm=True,
            max_seq_len=minibatch_size
        )
    )
)
