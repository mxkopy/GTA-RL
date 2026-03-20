import config
import numpy as np
import torch
import torch.nn as nn
from math import prod
from typing import Dict, Any, Optional
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import TARGET_NETWORK_ACTION_DIST_INPUTS, ValueFunctionAPI
from environment import VideoState

class UNet(nn.Module):

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    class Down(nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.MaxPool2d(2),
                UNet.double_conv(in_channels, out_channels)
            )
        
        def forward(self, x):
            return self.model(x)

    class Up(nn.Module):

        def __init__(self, in_channels, out_channels, bilinear=False):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = UNet.double_conv(in_channels, out_channels)
            else:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                self.conv = UNet.double_conv(out_channels * 2, out_channels)

        def forward(self, x, x_p):
            x = self.up(x)
            diffY = x_p.shape[-2] - x.shape[-2]
            diffX = x_p.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat((x_p, x), dim=-3)
            return self.conv(x)

    def __init__(self, architecture: list[int], in_channels=config.image_shape[0], out_channels=config.image_shape[0]):
        super().__init__()
        self.entry = self.double_conv(in_channels, architecture[0])
        self.down = nn.ModuleList([UNet.Down(i, o) for i, o in zip(architecture[:-1], architecture[1:])])
        self.up = nn.ModuleList([UNet.Up(o, i) for o, i in zip(architecture[:0:-1], architecture[-2::-1])])
        self.exit = nn.Conv2d(architecture[0], out_channels, 1, 1)

    def forward(self, x):
        X = [self.entry(x)]
        for layer in self.down:
            X += [layer(X[-1])]
        y = X.pop()
        for layer in self.up:
            y = layer(y, X.pop())
        return self.exit(y)

class UNet3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.voxel_model = nn.Sequential(
            # nn.Conv3d(config.image_shape[0], config.image_shape[0], 3, 1, padding='same'),
            # nn.Conv3d(config.image_shape[0], config.image_shape[0], 3, 1, padding='same'),
            nn.Flatten(),
            nn.Linear(config.voxel_depth * prod(config.image_shape), config.num_visual_features)
        )
        self.point_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(config.image_shape), config.num_visual_features)
        )
        self.combine = nn.Sequential(
            nn.Linear(config.num_visual_features * 2, config.num_visual_features * 2),
            nn.LeakyReLU(),
            nn.Linear(config.num_visual_features * 2, config.num_visual_features)
        )

    @staticmethod
    def fill_voxels(x):
        with torch.no_grad():
            device = x.device
            x = x.to(device='cuda')
            voxels = VideoState.voxelize(x)
            idxs = torch.arange(voxels.shape[-3], device=voxels.device, dtype=torch.int).reshape(1, -1, 1, 1).repeat(1, 1, voxels.shape[-2], voxels.shape[-1])
            voxels[idxs < (voxels * idxs).sum(dim=-3, keepdim=True)] = 1
            voxels = voxels.to(device=device)
            return voxels

    def forward(self, x):
        voxels = self.fill_voxels(x)
        point, voxel = self.point_model(x), self.voxel_model(voxels)
        combined = torch.cat((point, voxel), dim=-1)
        return self.combine(combined)

# Extracts visual features
# Should not care about frame stacking; takes (-1, image_size...) shaped input
class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            UNet3D(),
            nn.LeakyReLU(),
            nn.Linear(config.num_visual_features, config.num_visual_features)
        )

    def forward(self, images):
        return self.model(images)

# Embeds visual (and/or other) features as hidden/latent features 
# Does care about frame stacking; takes (batch, n_frames * num_features) shaped input
class Embedding(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Linear(config.num_features, config.num_features),
            nn.LeakyReLU(),
            nn.Linear(config.num_features, config.embedding_size)
        )

    def forward(self, features):
        return self.model(features)

# TODO: Add LSTM nn.Module to encapsulate the ugliness in Model.compute_embeddings_and_state_outs

# PPO actor model. 
# Produces mean & std values defining action probability distribution from hidden features
class Actor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Linear(config.embedding_size, 2*prod(config.action_space_shape)),
            nn.LeakyReLU(),
            nn.Linear(2*prod(config.action_space_shape), 2*prod(config.action_space_shape))
        )

    def forward(self, embedding):
        return self.model(embedding)


# PPO critic model. 
# Estimates the value of an action in a given state (hopefully encapsulated in the hidden features)
class Critic(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Linear(config.embedding_size + prod(config.action_space_shape), config.embedding_size + prod(config.action_space_shape)),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_size + prod(config.action_space_shape), 1),
        )

    def forward(self, embedding, action):
        state_action_pair = torch.cat((embedding, action), dim=-1)
        return self.model(state_action_pair)

class PPODriver(TorchRLModule, ValueFunctionAPI):
     
    @override(TorchRLModule)
    def setup(self):
        self.visual = VisualModel()
        self.lstm = nn.LSTM(config.num_features, config.num_features, num_layers=config.lstm_num_layers, batch_first=True)
        self.embedding = Embedding()
        self.actor = Actor()
        self.critic = Critic()

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(shape=(self.lstm.num_layers, self.lstm.hidden_size), dtype=np.float32),
            "c": np.zeros(shape=(self.lstm.num_layers, self.lstm.hidden_size), dtype=np.float32)
        }

    def compute_embeddings_and_state_outs(self, batch: Dict[str, Any]):
        images = batch[Columns.OBS].reshape(-1, *config.image_shape)
        image_features = self.visual(images)
        hidden_features = image_features.reshape(-1, config.num_features)
        # embeddings will have a shape of the form (batch_size * num_batches, embedding_size)
        # Hence embeddings.shape[-2] // batch_size is the size of the batch dimension
        # And max_seq_len is the sequence length (since everything gets padded to it)
        hidden_features = hidden_features.reshape(-(hidden_features.shape[-2] // -self.model_config['max_seq_len']), -1, hidden_features.shape[-1])
        # The hidden states are shaped (batch, numlayers, x), but lstms take the batch second for hidden states
        h, c = batch[Columns.STATE_IN]['h'], batch[Columns.STATE_IN]['c']
        h, c = torch.transpose(h, 0, 1).contiguous(), torch.transpose(c, 0, 1).contiguous()
        hidden_features, (h, c) = self.lstm(hidden_features, (h, c))
        h, c = torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)
        embeddings = self.embedding(hidden_features)
        return embeddings, {'h': h, 'c': c}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None, **kwargs):
        if embeddings is None:
            embeddings, _ = self.compute_embeddings_and_state_outs(batch)
        values = self.critic(embeddings, batch[Columns.ACTIONS])
        return values.reshape(*batch[Columns.LOSS_MASK].shape)
    
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        embeddings, states_out = self.compute_embeddings_and_state_outs(batch)
        logits = self.actor(embeddings)
        logits = logits.reshape(-(logits.shape[-2] // -self.model_config['max_seq_len']), -1, logits.shape[-1])
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: states_out,
            Columns.EMBEDDINGS: embeddings
        }


import config
from environment import Environment
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

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
        rl_module_spec=RLModuleSpec(
            module_class=PPODriver,
            observation_space=Environment().observation_space,
            action_space=Environment().action_space,
            model_config={
                'max_seq_len': config.minibatch_size,
            }
        )
    )
)
