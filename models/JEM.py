import torch as t
import torch.nn as nn
from accelerate import Accelerator

from DataModule import DataModule
from models.wideresnet import Wide_ResNet
from models.WideResNetYOPO import WideResNetYOPO


class F(nn.Module):
    def __init__(self, depth: int, width: int, norm: str | None, n_classes: int, n_channels: int, model: str, **config):
        super(F, self).__init__()
        self.f = WideResNetYOPO(depth, width, n_channels, norm=norm) if model == "yopo" else Wide_ResNet(depth, width, norm=norm)
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


def get_model_and_buffer(datamodule: DataModule, buffer_size: int, load_path: str = None, accelerator: Accelerator = None, **config):
    """
    Gets the model and the replay buffer.

    If using multiple GPUs and BatchNorm, convert BatchNorm layers to SyncBatchNorm.
    """
    f = F(n_channels=datamodule.img_shape[0], n_classes=datamodule.n_classes, **config)

    if accelerator and config["norm"] == "batch":
        accelerator.print(f"Using {t.cuda.device_count()} GPUs. Converting BatchNorm to SyncBatchNorm.")
        f = nn.SyncBatchNorm.convert_sync_batchnorm(f)

    if load_path:
        f.load_state_dict(t.load(load_path)["model_state_dict"])

    if accelerator:
        f = accelerator.prepare(f)

    return f, init_random(datamodule.img_shape, buffer_size)


def get_optimizer(f: nn.Module, load_path: str = None, accelerator: Accelerator = None, **config):
    """
    Initializes optimizer.
    """
    params = f.class_output.parameters() if config["clf_only"] else f.parameters()

    if config["optimizer"] == "adam":
        optim = t.optim.Adam(params, config["lr"], betas=(0.9, 0.999), weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optim = t.optim.SGD(params, config["lr"], momentum=0.9, weight_decay=config["weight_decay"])

    if load_path:
        optim.load_state_dict(t.load(load_path)["optimizer_state_dict"])

    if accelerator:
        optim = accelerator.prepare(optim)

    return optim
