n_frames = 1

observation_space_shape = {
    'image': (5, 360, 640),
    'velocity': (3,),
}

action_space_shape = (3,)

visual_channels = [observation_space_shape['image'][0], 3, 3, 3, 3]
visual_embedding_size = (8, 8)
embedding_size = 16

device = 'cuda'
