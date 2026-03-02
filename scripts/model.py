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


def unflatten_batch(flattened):
    flat_images = flattened[..., :prod(config.image_shape)]
    flat_velocities = flattened[..., prod(config.image_shape):]
    images = flat_images.reshape(-1, *config.image_shape)
    velocities = flat_velocities.reshape(-1, *config.velocity_shape)
    return images, velocities

class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(config.image_shape[0], 64, 7, 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, config.visual_embedding_size),
            nn.ReLU(),
            nn.Linear(config.visual_embedding_size, config.visual_embedding_size)        
        )

    # input is of shape (..., n_frames)
    def forward(self, images):
        return self.model(images)
    
class Embedding(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = (config.visual_embedding_size + prod(config.velocity_shape)) * config.n_frames
        self.embedding = nn.Sequential(
            nn.Linear(self.hidden_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

    def forward(self, visual_embedding, velocity):
        visual_embedding = visual_embedding.reshape(-1, config.n_frames * config.visual_embedding_size)
        velocity = velocity.reshape(-1, config.n_frames * np.prod(config.velocity_shape))
        features = torch.cat((visual_embedding, velocity), dim=1)
        return self.embedding(features)
    
class Actor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actor = nn.Sequential(
            nn.Linear(config.embedding_size, prod(config.action_space_shape)),
            nn.ReLU(),
            nn.Linear(prod(config.action_space_shape), 2*prod(config.action_space_shape)),
            nn.ReLU(),
            nn.Linear(2*prod(config.action_space_shape), 2*prod(config.action_space_shape))
        )

    def forward(self, embedding):
        return self.actor(embedding)

class Critic(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = nn.Sequential(
            nn.Linear(config.embedding_size, prod(config.action_space_shape)),
            nn.ReLU(),
            nn.Linear(prod(config.action_space_shape), 2*prod(config.action_space_shape)),
            nn.ReLU(),
            nn.Linear(2*prod(config.action_space_shape), 1)
        )

    def forward(self, embedding):
        return self.value(embedding)


class Model(TorchRLModule, ValueFunctionAPI):
     
    @override(TorchRLModule)
    def setup(self, **kwargs):
        self.visual = VisualModel()
        self.embedding = Embedding()
        self.actor = Actor()
        self.critic = Critic()

    def compute_embeddings(self, batch: Dict[str, Any]):
        images, velocities = unflatten_batch(batch[Columns.OBS])
        image_features = self.visual(images)
        embeddings = self.embedding(image_features, velocities)
        return embeddings

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None, **kwargs):
        if embeddings is None:
            images, velocities = unflatten_batch(batch[Columns.OBS])
            images = self.visual(images)
            embeddings = self.embedding(images, velocities)
        return self.critic(embeddings)
    
    def compute_embeddings_and_logits(self, batch):
        images, velocities = unflatten_batch(batch[Columns.OBS])
        vis = self.visual(images)
        embeddings = self.embedding(vis, velocities)
        logits = self.actor(embeddings)
        return (
            embeddings,
            logits
        )
    
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        _, logits = self.compute_embeddings_and_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings, logits = self.compute_embeddings_and_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }
