import config
import torch
import torch.nn as nn
from math import prod, floor, log

class ResidualBlock(nn.Module):

    @staticmethod
    def double_conv(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __init__(self, in_channels, out_channels, middle_channels=None, kernel_size=3):
        super().__init__()
        if middle_channels is None:
            if out_channels >= 2*in_channels:
                middle_channels = out_channels - in_channels
            else:
                middle_channels = out_channels
        self.layer = self.double_conv(in_channels, middle_channels, kernel_size=kernel_size)
        self.res = nn.Conv2d(in_channels + middle_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        y = self.layer(x)
        y = torch.cat((x, y), dim=-3)
        y = self.res(y)
        return y

class VisualModel(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.model = nn.Sequential(
            ResidualBlock(config.image_shape[0], 2),
            nn.AvgPool2d(2),
            nn.ReLU(),

            ResidualBlock(2, 4),
            nn.AvgPool2d(2),
            nn.ReLU(),

            ResidualBlock(4, 8),
            nn.AvgPool2d(2),
            nn.ReLU(),

            ResidualBlock(8, 16),
            nn.AvgPool2d(2),
            nn.ReLU(),
            
            ResidualBlock(16, 32),
            nn.AvgPool2d(2),
            nn.ReLU(),
            
            ResidualBlock(32, 64),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )
        T = torch.zeros(config.image_shape).unsqueeze(0)
        S = prod(self.model(T).shape)
        H = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )
        self.model.extend(H)


    def forward(self, x):
        return self.model(x)