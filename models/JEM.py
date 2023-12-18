import torch as t
import torch.nn as nn
from DataModule import DataModule
from models import wideresnet
from models.WideResNetYOPO import WideResNetYOPO


from accelerate import Accelerator


class F(nn.Module):
    def __init__(
        self,
        depth: int,
        width: int,
        norm: str | None,
        dropout_rate: float,
        n_classes: int,
        n_channels: int,
        model: str,
        **config,
    ):
        super(F, self).__init__()
        if model == "yopo":
            self.f = WideResNetYOPO(depth, width, n_channels, norm=norm)
        else:
            self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)

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


def init_random(img_shape: tuple[int, int, int], buffer_size: int):
    n_channels, img_size, _ = img_shape
    return t.FloatTensor(buffer_size, n_channels, img_size, img_size).uniform_(-1, 1)


def get_model_and_buffer(accelerator: Accelerator, datamodule: DataModule, buffer_size: int, load_path: str = None, **config):
    f = F(n_channels=datamodule.img_shape[0], n_classes=datamodule.n_classes, **config)

    """If using multiple GPUs and BatchNorm, convert BatchNorm layers to SyncBatchNorm."""

    if t.cuda.device_count() > 1 and config["norm"] == "batch":
        accelerator.print(f"Using {t.cuda.device_count()} GPUs. Converting BatchNorm to SyncBatchNorm.")
        f = nn.SyncBatchNorm.convert_sync_batchnorm(f)

    if load_path is not None:
        f.load_state_dict(t.load(load_path)["model_state_dict"])

    if t.cuda.device_count() > 1:
        f = accelerator.prepare(f)

    return f, init_random(datamodule.img_shape, buffer_size)


def get_optimizer(accelerator: Accelerator, f: nn.Module, load_path: str = None, **config):
    """Initialize optimizer"""
    params = f.class_output.parameters() if config["clf_only"] else f.parameters()

    if config["optimizer"] == "adam":
        optim = t.optim.Adam(params, config["lr"], betas=(0.9, 0.999), weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optim = t.optim.SGD(params, config["lr"], momentum=0.9, weight_decay=config["weight_decay"])
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented.")

    if load_path is not None:
        optim.load_state_dict(t.load(load_path)["optimizer_state_dict"])

    if t.cuda.device_count() > 1:
        optim = accelerator.prepare(optim)

    return optim
