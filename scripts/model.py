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

# Extracts visual features
# Loosely based on the first N layers of a YOLO segmentation model
# Does not care about frame stacking; takes (-1, image_size...) shaped input
class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(config.image_shape),
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

# Embeds visual (and/or other) features as hidden/latent features 
# Does care about frame stacking; takes (batch, n_frames * num_features) shaped input
# Output does not care about frame stacking 
class Embedding(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            # nn.LayerNorm(config.num_features),
            nn.Linear(config.num_features, config.embedding_size),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

    def forward(self, features):
        return self.model(features)


# class Recurrent(nn.Module):

#     def __init__(self, *args, **kwargs):
#         self.model = nn.LSTM(*args, **kwargs)

#     def forward(self, batch: Dict[str, Any]):
#         pass

# TODO: Add LSTM nn.Module to encapsulate the ugliness in Model.compute_embeddings_and_state_outs

# PPO actor model. 
# Produces mean & std values defining action probability distribution from hidden features
class Actor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.LeakyReLU(),
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
            nn.Linear(config.embedding_size + prod(config.action_space_shape), config.embedding_size + prod(config.action_space_shape)),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_size + prod(config.action_space_shape), config.embedding_size + prod(config.action_space_shape)),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_size + prod(config.action_space_shape), 1),
        )

    def forward(self, embedding, action):
        state_action_pair = torch.cat((embedding, action), dim=-1)
        return self.model(state_action_pair)



class Model(TorchRLModule, ValueFunctionAPI):
     
    @override(TorchRLModule)
    def setup(self):
        self.visual = VisualModel()
        self.lstm = nn.LSTM(config.num_features, config.num_features, num_layers=4, batch_first=True)
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
    def unflatten_batch(flattened):
        images = flattened.reshape(-1, *config.image_shape)
        return images

    def compute_embeddings_and_state_outs(self, batch: Dict[str, Any]):
        images = self.unflatten_batch(batch[Columns.OBS])
        image_features = self.visual(images)
        hidden_features = image_features.reshape(-1, config.num_features)
        h, c = batch[Columns.STATE_IN]['h'], batch[Columns.STATE_IN]['c']
        # The hidden states are shaped (batch, numlayers, x), but lstms take the batch second for hidden states
        h, c = torch.transpose(h, 0, 1).contiguous(), torch.transpose(c, 0, 1).contiguous()
        # embeddings will have a shape of the form (batch_size * num_batches, embedding_size)
        # Hence embeddings.shape[-2] // batch_size is the size of the batch dimension
        # And max_seq_len is the sequence length (since everything gets padded to it)
        hidden_features = hidden_features.reshape(-(hidden_features.shape[-2] // -self.model_config['max_seq_len']), -1, hidden_features.shape[-1])
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

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.typing import ModuleID, TensorType

# Custom learner adding L1 weight regularization
class Learner(PPOTorchLearner):

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:

        base_total_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        # Compute the mean of all the RLModule's weights' absolute values.
        parameters = self.get_parameters(self.module[module_id])
        mean_abs_weight = torch.mean(torch.cat([p.reshape(-1).abs() for p in parameters]))

        self.metrics.log_value(
            key=(module_id, "mean_abs_weight"),
            value=mean_abs_weight,
            window=1,
        )

        total_loss = (
            base_total_loss
            + config.learner_config_dict["regularizer_coeff"] * mean_abs_weight
        )

        return total_loss