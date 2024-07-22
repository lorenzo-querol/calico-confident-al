from argparse import Namespace

import torch as t
import torch.nn as nn

from DataModule import MedMNISTDataModule
from models.Wide_ResNet_YOPO import Wide_ResNet


class F(nn.Module):
    def __init__(self, depth: int, width: int, n_classes: int, n_channels: int, norm: str | None):
        super(F, self).__init__()
        self.f = Wide_ResNet(depth, width, n_channels, norm=norm)
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


def get_optim(f: nn.Module, args: Namespace):
    optimizers = {"adam": t.optim.Adam, "sgd": t.optim.SGD}

    if args.optim not in optimizers:
        raise ValueError(f"Optimizer {args.optim} not supported. Supported optimizers: {list(optimizers.keys())}")

    base_optim = optimizers[args.optim]

    optim_params = {"params": f.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}

    if args.optim == "sgd":
        optim_params["momentum"] = 0.9

    return base_optim(**optim_params)


def get_model(datamodule: MedMNISTDataModule, args: Namespace, ckpt_path: str | None = None):
    f = F(args.depth, args.width, datamodule.n_classes, datamodule.img_shape[0], args.norm)

    if ckpt_path is not None:
        print(f"Loading model from {ckpt_path}")
        f.load_state_dict(t.load(ckpt_path)["model_state_dict"])

    return f
