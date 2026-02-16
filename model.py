import config
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import TARGET_NETWORK_ACTION_DIST_INPUTS, ValueFunctionAPI

class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(config.observation_space_shape['image'][0], 64, 7, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, stride=2),
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

    def forward(self, img):
        return self.model(img)
    

# vis = VisualModel()
# out = vis(torch.rand( (1, *config.observation_space_shape['image']) ))


class Embedding(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.visual_embedding_size + config.observation_space_shape['velocity'][0]
        self.embedding = nn.Sequential(
            nn.Linear(self.hidden_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

    def forward(self, visual_embedding, velocity):
        visual_embedding = visual_embedding.reshape(-1, config.visual_embedding_size)
        velocity = velocity.reshape(-1, np.prod(config.observation_space_shape['velocity']))
        features = torch.cat((visual_embedding, velocity), dim=1)
        return self.embedding(features)
    
class Actor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actor = nn.Sequential(
            nn.Linear(config.embedding_size, config.action_space_shape[0]),
            nn.ReLU(),
            nn.Linear(config.action_space_shape[0], 2*config.action_space_shape[0]),
            nn.ReLU(),
            nn.Linear(2*config.action_space_shape[0], 2*config.action_space_shape[0])
        )

    def forward(self, embedding):
        return self.actor(embedding)

class Critic(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = nn.Sequential(
            nn.Linear(config.embedding_size, config.action_space_shape[0]),
            nn.ReLU(),
            nn.Linear(config.action_space_shape[0], 2*config.action_space_shape[0]),
            nn.ReLU(),
            nn.Linear(2*config.action_space_shape[0], 1)
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

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None, **kwargs):
        if embeddings is None:
            img, vel = batch[Columns.OBS]
            vis = self.visual(img)
            embeddings = self.embedding(vis, vel)
        return self.critic(embeddings)
    
    def compute_embeddings_and_logits(self, batch):
        img, vel = batch[Columns.OBS]
        vis = self.visual(img)
        embeddings = self.embedding(vis, vel)
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
