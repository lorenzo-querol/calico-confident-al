import torch as t
import torch.nn as nn
from models import wideresnet
from models import wideresnet_yopo

im_sz = 32
n_ch = 3


class F(nn.Module):
    def __init__(self, depth, width, norm, dropout_rate, n_classes, n_channels, model):
        super(F, self).__init__()
        self.norm = norm

        if model == "yopo":
            self.f = wideresnet_yopo.Wide_ResNet(depth, width, n_channels, norm=norm)
        else:
            self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)

        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def feature(self, x):
        penult_z = self.f(x, feature=True)
        return penult_z

    def forward(self, x, y=None):
        penult_z = self.f(x, feature=True)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x, feature=True)
        output = self.class_output(penult_z).squeeze()
        return output


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, model="wrn", args=None):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes, model=model, args=args)

    def forward(self, x, y=None):
        logits = self.classify(x)

        if y is None:
            v = logits.logsumexp(1)
            # print("log sum exp", v)
            return v
        else:
            return t.gather(logits, 1, y[:, None])


def init_random(datamodule, bs):
    n_channels = datamodule.img_shape[0]
    im_size = datamodule.img_shape[1]

    return t.FloatTensor(bs, n_channels, im_size, im_size).uniform_(-1, 1)


def get_model_and_buffer(accelerator, datamodule, uncond, depth, width, buffer_size, dropout_rate, model, norm, **config):
    model_cls = F if uncond else CCF
    f = model_cls(depth, width, norm, dropout_rate, datamodule.n_classes, datamodule.img_shape[0], model)

    """If using multiple GPUs, convert BatchNorm layers to SyncBatchNorm since WRN was enabled"""
    if t.cuda.device_count() > 1:
        accelerator.print(f"Using {t.cuda.device_count()} GPUs. Converting BatchNorm to SyncBatchNorm.")
        f = nn.SyncBatchNorm.convert_sync_batchnorm(f)

    if not uncond:
        assert buffer_size % datamodule.n_classes == 0, "Buffer size must be divisible by n_classes"

    replay_buffer = init_random(datamodule, buffer_size)

    return f, replay_buffer
