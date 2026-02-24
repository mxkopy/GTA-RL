from math import prod

n_frames = 1


image_shape = (4, 360, 640) 
velocity_shape = (3,)
observation_space_shape = ((prod(image_shape) + prod(velocity_shape)),)

action_space_shape = (3,)

visual_channels = [image_shape[0], 3, 3, 3, 3]
visual_embedding_size = 64
embedding_size = 16

device = 'cuda'
