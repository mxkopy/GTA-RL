from math import prod

# Observation & Action spaces
n_frames = 1
image_shape = (1, 360, 640)
observation_space_shape = (n_frames * prod(image_shape),)
action_space_shape = (2,)

# Model architecture
num_visual_features = 64
num_features = num_visual_features * n_frames
embedding_size = 64
device = 'cuda'
