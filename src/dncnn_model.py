
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Deep convolutional neural network for image denoising.
    Predicts noise and subtracts it from the noisy input.
    """

    def __init__(self, depth=17, channels=64):
        super().__init__()

        layers = [
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        ]

        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ])

        layers.append(
            nn.Conv2d(channels, 3, 3, padding=1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)
