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
    flattened = flattened.reshape(-1, prod(config.observation_space_shape) // config.n_frames)
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
            nn.Linear(512, config.num_visual_features),
            nn.ReLU(),
            nn.Linear(config.num_visual_features, config.num_visual_features)        
        )

    def forward(self, images):
        return self.model(images)




class Embedding(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.ReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

    def forward(self, hidden_features):
        return self.embedding(hidden_features)



# TODO: Add LSTM nn.Module to encapsulate the ugliness in Model.compute_embeddings_and_state_outs


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
    def setup(self):
        self.visual = VisualModel()
        self.lstm = torch.nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, batch_first=True)
        self.embedding = Embedding()
        self.actor = Actor()
        self.critic = Critic()

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(shape=(self.lstm.num_layers, self.lstm.hidden_size), dtype=np.float32),
            "c": np.zeros(shape=(self.lstm.num_layers, self.lstm.hidden_size), dtype=np.float32)
        }

    @staticmethod
    def concat_features(visual_features, velocities):
        visual_features = visual_features.reshape(-1, config.n_frames * config.num_visual_features)
        velocities = velocities.reshape(-1, config.n_frames * np.prod(config.velocity_shape))
        return torch.cat((visual_features, velocities), dim=-1)

    def compute_embeddings_and_state_outs(self, batch: Dict[str, Any]):
        images, velocities = unflatten_batch(batch[Columns.OBS])
        image_features = self.visual(images)
        hidden_features = self.concat_features(image_features, velocities)
        h, c = batch[Columns.STATE_IN]['h'], batch[Columns.STATE_IN]['c']
        # The hidden states are shaped (batch, numlayers, x), but lstms take the batch second for hidden states
        h, c = torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)
        # embeddings will have a shape of the form (batch_size * num_batches, embedding_size)
        # Hence embeddings.shape[-2] // batch_size is the size of the batch dimension
        # Since everything gets padded up to max_seq_len we really want minibatch_size == max_seq_len in the AlgorithmConfig
        hidden_features = hidden_features.reshape(-(hidden_features.shape[-2] // -self.model_config['max_seq_len']), -1, hidden_features.shape[-1])
        hidden_features, (h, c) = self.lstm(hidden_features, (h, c))
        h, c = torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)
        embeddings = self.embedding(hidden_features)
        return embeddings, {'h': h, 'c': c}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None, **kwargs):
        if embeddings is None:
            embeddings, _ = self.compute_embeddings_and_state_outs(batch)
        values = self.critic(embeddings)
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
