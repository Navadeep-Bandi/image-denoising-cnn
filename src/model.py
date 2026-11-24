
import torch
import torch.nn as nn


class DenoisingCNN(nn.Module):

    def __init__(self):
        super(DenoisingCNN, self).__init__()

        # Encoder layers
        self.enc1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.enc2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Decoder layers
        self.dec1 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.dec2 = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))

        x = self.relu(self.dec1(x))
        x = self.sigmoid(self.dec2(x))

        return x


if __name__ == "__main__":

    model = DenoisingCNN()

    dummy_input = torch.rand(1, 3, 128, 128)

    output = model(dummy_input)

    print("Model architecture:")
    print(model)

    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters())
    )
