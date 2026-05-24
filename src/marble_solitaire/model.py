import torch
import torch.nn as nn


class SolitaireNet(nn.Module):
    """
    Dual-headed CNN for marble solitaire.
    Input: (B, 2, 7, 7) - channel 0: marbles, channel 1: valid mask
    Output: (policy [B, 196], value [B, 1])
    """

    def __init__(self, channels: int = 64, n_blocks: int = 3):
        super().__init__()
        layers = [nn.Conv2d(2, channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(channels), nn.ReLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(channels), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Conv2d(channels, 4, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc = nn.Sequential(
            nn.Linear(49, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        trunk_out = self.trunk(x)
        policy = self.policy_head(trunk_out).view(x.size(0), -1)
        value = self.value_conv(trunk_out).view(x.size(0), -1)
        value = self.value_fc(value)
        return policy, value
