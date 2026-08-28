import torch
import torch.nn as nn


class FashionMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = FashionMLP()

    print(model)

    x = torch.randn(64, 1, 28, 28)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)