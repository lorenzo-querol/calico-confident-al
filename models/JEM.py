import torch.nn as nn

from models.WideResNet import WideResNet


class F(nn.Module):
    def __init__(self, depth: int, width: int, n_classes: int, n_channels: int):
        super(F, self).__init__()
        self.f = WideResNet(depth, width, 0.0, n_channels)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def feature(self, x):
        z = self.f(x)
        return z

    def forward(self, x):
        z = self.f(x)
        return self.energy_output(z).squeeze()

    def classify(self, x):
        z = self.f(x)
        return self.class_output(z).squeeze()
