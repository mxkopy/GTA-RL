import config
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import (
    TARGET_NETWORK_ACTION_DIST_INPUTS,
    ValueFunctionAPI
)

class VisualModel(nn.Module):

    def __init__(self, visual_channels=config.visual_channels, visual_embedding_size=config.visual_embedding_size):
        super().__init__()
        self.model = nn.Sequential()
        for in_channels, out_channels in zip(visual_channels, visual_channels[1:]):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), padding='same'),
                nn.ReLU()
            )
            self.model.append(conv_layer)
        self.model.append(nn.Conv2d(visual_channels[-1], 1, (1, 1)))
        self.model.append(nn.AdaptiveAvgPool2d(visual_embedding_size))

    def forward(self, img):
        return self.model(img)

    def jit(self):
        device = list(self.model.parameters())[0].device
        dynamic_shapes = torch.export.ShapesCollection()
        batched_image = torch.rand(2, *config.observation_space_shape['image']).to(device)
        dynamic_shapes[batched_image] = { 0: torch.export.Dim.DYNAMIC }
        return torch.export.export(
            self,
            args=(batched_image,),
            dynamic_shapes=dynamic_shapes
        ).module()



class Embedding(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = np.prod(config.visual_embedding_size) + config.observation_space_shape['velocity'][0]
        self.embedding = nn.Sequential(
            nn.Linear(self.hidden_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

    def forward(self, visual_embedding, velocity):
        visual_embedding = visual_embedding.reshape(-1, np.prod(config.visual_embedding_size))
        velocity = velocity.reshape(-1, np.prod(config.observation_space_shape['velocity']))
        features = torch.cat((visual_embedding, velocity), dim=1)
        return self.embedding(features)
    
    def jit(self):
        device = list(self.embedding.parameters())[0].device
        dynamic_shapes = torch.export.ShapesCollection()
        batched_visual_embedding = torch.rand(2, 1, *config.visual_embedding_size).to(device)
        batched_velocity = torch.rand(2, *config.observation_space_shape['velocity']).to(device)
        dynamic_shapes[batched_visual_embedding] = { 0: torch.export.Dim.DYNAMIC }
        dynamic_shapes[batched_velocity] = { 0: torch.export.Dim.DYNAMIC }
        return torch.export.export(
            self,
            args=(batched_visual_embedding, batched_velocity),
            dynamic_shapes=dynamic_shapes
        ).module()


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

    def jit(self):
        device = list(self.actor.parameters())[0].device
        dynamic_shapes = torch.export.ShapesCollection()
        batched = torch.rand(2, config.embedding_size).to(device)
        dynamic_shapes[batched] = { 0: torch.export.Dim.DYNAMIC }
        return torch.export.export(
            self,
            args=(batched,),
            dynamic_shapes=dynamic_shapes
        ).module()

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

    def jit(self):
        device = list(self.value.parameters())[0].device
        dynamic_shapes = torch.export.ShapesCollection()
        batched = torch.rand(2, config.embedding_size).to(device)
        dynamic_shapes[batched] = { 0: torch.export.Dim.DYNAMIC }
        return torch.export.export(
            self,
            args=(batched,),
            dynamic_shapes=dynamic_shapes
        ).module()

class Model(TorchRLModule, ValueFunctionAPI):
     
    @override(TorchRLModule)
    def setup(self, **kwargs):
        test_input = self.observation_space.sample()
        test_output = self.action_space.sample()
        self.visual = VisualModel().to(device='cuda')#.jit()
        self.embedding = Embedding().to(device='cuda')#.jit()
        self.actor = Actor().to(device='cuda')#.jit()
        self.critic = Critic().to(device='cuda')#.jit()

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
