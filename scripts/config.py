from math import prod

# Observation & Action spaces
n_frames=1
voxel_depth=16
# image_shape=(1, 360, 640)
image_shape=(1, 90, 160)
observation_space_shape=(n_frames * prod(image_shape),)
action_space_shape=(2,)
depth_cutoff=20