import config
from typing import Dict, Any, Optional
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import (
    TARGET_NETWORK_ACTION_DIST_INPUTS,
    ValueFunctionAPI
)

torch, nn = try_import_torch()

class Model(TorchRLModule, ValueFunctionAPI):
     
    @override(TorchRLModule)
    def setup(self, **kwargs):
        test_input = self.observation_space.sample()
        test_output = self.action_space.sample()
        flattened_input_size = test_input[0].size + test_input[1].size
        flattened_output_size = test_output.size
        embedding_size = 128
        self.embedding = nn.Linear(flattened_input_size, embedding_size).to(device='cuda')
        self.value = nn.Linear(embedding_size, 1).to(device='cuda')
        self.action = nn.Linear(embedding_size, flattened_output_size).to(device='cuda')
        
    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None, **kwargs):
        if embeddings is None:
            img, vel = batch[Columns.OBS]
            img = img.reshape(-1, torch.numel(img[0, ...]))
            obs = torch.cat((img, vel), dim=1)
            embeddings = self.embedding(obs)
        return self.value(embeddings)
    
    def compute_embeddings_and_logits(self, batch):
        img, vel = batch[Columns.OBS]
        img = img.reshape(-1, torch.numel(img[0, ...]))
        obs = torch.cat((img, vel), dim=1)
        embeddings = self.embedding(obs)
        logits = self.action(embeddings)
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


# TODO: This component of the actor/critic model should be trained separately & be otherwise static
# Otherwise, it might take too long to train the rest of the network 
# What it really should do is provide per-pixel kinematic information like distance & velocity relative to the viewer


# class VisualModel(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential()
#         for in_channels, out_channels in zip(config.visual_channels, config.visual_channels[1:]):
#             conv_layer = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, (3, 3), padding='same'),
#                 nn.ReLU()
#             )
#             self.model.append(conv_layer)
#         self.model.append(nn.AdaptiveAvgPool2d(config.visual_features_size))

#     def forward(self, img):
#         return self.model(img)

# class DriverModelBase(nn.Module):

#     def __init__(self, distribution: None | type[torch.distributions.Distribution] = None):
#         super().__init__()
#         self.visual = VisualModel()
#         self.hidden_size = config.visual_features_size[0] * config.visual_features_size[1] * config.visual_channels[-1] + 1
#         self.collate_mean = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, config.action_sizes['controller'][0])
#         )
#         self.collate_std = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, config.action_sizes['controller'][0])
#         )
#         self.distribution = distribution

#     def forward(self, state):

#         features = self.visual(state.image)
#         features = features.reshape(features.size(0), -1)
#         features = torch.cat((features, torch.square(state.velocity).sum(dim=1, keepdim=True).sqrt()), dim=1)
#         if self.distribution is None:
#             return self.collate_mean(features)
#         else:
#             return self.distribution(self.collate_mean(features), self.collate_std(features)).sample()

# class DriverActorModel(DriverModelBase):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def forward(self, state):
#         action = super().forward(state)
#         action[:, 0:2] = torch.tanh(action[:, 0:2])
#         return action

#     def jit(self):
#         device = list(self.visual.parameters())[0].device
#         state = State.rand(batch_size=2).to(device=device)
#         return torch.export.export(
#             self,
#             args=(state,),
#             dynamic_shapes=state.dynamic_shapes()
#         ).module()

# class DriverCriticModel(DriverModelBase):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.value_function = nn.Sequential(
#             nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size + config.action_sizes['controller'][0], 1)
#         )

#     def forward(self, state: State, action: Action):
#         features = self.visual(state.image)
#         features = features.reshape(features.size(0), -1)
#         features = torch.cat((features, torch.square(state.velocity).sum(dim=1, keepdim=True).sqrt()), dim=1)
#         value = self.value_function(torch.cat((features, action), dim=1))
#         return value

#     def jit(self):
#         dynamic_shapes = torch.export.ShapesCollection()
#         device = list(self.visual.parameters())[0].device
#         state = State.rand(batch_size=2).to(device=device)
#         action = torch.rand(2, *config.action_sizes['controller']).to(device=device)
#         args = (state, action)
#         args[0].dynamic_shapes(dynamic_shapes)
#         dynamic_shapes[action] = {0: torch.export.Dim.DYNAMIC}
#         return torch.export.export(
#             self,
#             args=args,
#             dynamic_shapes=dynamic_shapes
#         ).module()