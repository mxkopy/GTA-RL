from math import prod

# Observation & Action spaces
n_frames=1
voxel_depth=16
image_shape=(1, 360, 640)
observation_space_shape=(n_frames * prod(image_shape),)
action_space_shape=(2,)
depth_cutoff=25

# Model architecture
num_visual_features=16
num_features=num_visual_features * n_frames
embedding_size=16
lstm_num_layers=4
device = 'cuda'

# Training hyperparameters
learning_rate=1e-5
num_epochs=1
train_batch_size=128
minibatch_size=32
horizon=train_batch_size * 8
gae_lambda=0.95
clip_param=0.2
entropy_coeff=1e-5
vf_loss_coeff=1
lasso_coeff=1e-5
