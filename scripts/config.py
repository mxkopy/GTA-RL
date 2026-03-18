from math import prod

# Observation & Action spaces
n_frames = 1
image_shape = (1, 360, 640)
observation_space_shape = (n_frames * prod(image_shape),)
action_space_shape = (2,)

# Model architecture
num_visual_features = 64
num_features = num_visual_features * n_frames
embedding_size = 8
device = 'cuda'

# Training hyperparameters
learning_rate=1e-5
num_epochs=1
train_batch_size=128
minibatch_size=32
horizon=256
gae_lambda=0.9
clip_param=0.2
entropy_coeff=1e-6
vf_loss_coeff=1
lasso_coeff=1e-5
