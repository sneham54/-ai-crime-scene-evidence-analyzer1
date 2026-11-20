"""Compact genuine 3D DenseNet feature extractor."""
import torch
from torch import nn


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat((x, self.layer(x)), dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(layers):
            self.layers.append(DenseLayer3D(channels, growth_rate))
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition3D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )


class DenseNet3D(nn.Module):
    """Accepts B,C,T,H,W and returns B,C,T,H',W' with dense connectivity."""
    def __init__(self, in_channels=3, growth_rate=12, block_layers=(2, 2, 2), initial_features=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, initial_features, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(initial_features), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        channels = initial_features
        stages = []
        for index, layer_count in enumerate(block_layers):
            block = DenseBlock3D(channels, layer_count, growth_rate)
            stages.append(block)
            channels = block.out_channels
            if index != len(block_layers) - 1:
                compressed = max(8, channels // 2)
                stages.append(Transition3D(channels, compressed))
                channels = compressed
        stages.extend((nn.BatchNorm3d(channels), nn.ReLU(inplace=True)))
        self.features = nn.Sequential(*stages)
        self.out_channels = channels

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("DenseNet3D input must have shape (batch, channels, time, height, width)")
        return self.features(self.stem(x))
