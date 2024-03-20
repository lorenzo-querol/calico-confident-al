import torch as t
import torch.nn as nn

from DataModule import DataModule
from models.WideResNet import WideResNet
from models.WideResNetYOPO import WideResNetYOPO


class F(nn.Module):
    def __init__(self, depth: int, width: int, norm: str | None, n_classes: int, n_channels: int, model: str, **config):
        super(F, self).__init__()
        self.f = WideResNetYOPO(depth, width, n_channels, norm=norm) if model == "yopo" else WideResNet(depth, width, n_channels, norm=norm)
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


def get_optim(model: nn.Module, args):
    optimizers = {"adam": t.optim.Adam, "sgd": t.optim.SGD}

    if args.optim not in optimizers:
        raise ValueError(f"Optimizer {args.optim} not supported. Supported optimizers: {list(optimizers.keys())}")

    base_optim = optimizers[args.optim]

    optim_params = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}

    if args.optim == "sgd":
        optim_params["momentum"] = 0.9

    return base_optim(**optim_params)


def get_model(datamodule: DataModule, args, ckpt_path: str | None = None):
    f = F(args.depth, args.width, args.norm, datamodule.n_classes, datamodule.img_shape[0], args.model)

    if ckpt_path is not None:
        print(f"Loading model from {ckpt_path}.")
        f.load_state_dict(t.load(ckpt_path))

    return f
