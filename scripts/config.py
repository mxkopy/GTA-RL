from math import prod

n_frames = 1
image_shape = (1, 360, 640)
velocity_shape = (3,)
observation_space_shape = (n_frames * (prod(image_shape) + prod(velocity_shape)),)
action_space_shape = (2,)

visual_channels = [image_shape[0], 3, 3, 3, 3]
num_visual_features = 64
embedding_size = 16
hidden_size = (num_visual_features + prod(velocity_shape)) * n_frames

device = 'cuda'
