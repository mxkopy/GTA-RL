n_frames = 1

observation_space_shape = {
    'image': (7, 360, 640),
    'velocity': (3,),
}

action_space_shape = (3,)

visual_channels = [observation_space_shape['image'][0], 3, 3, 3, 3]
visual_embedding_size = 64
embedding_size = 16

device = 'cuda'
