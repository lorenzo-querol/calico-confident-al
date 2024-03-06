import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, norm, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes) if norm == "batch" else Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else Identity()
        self.bn2 = nn.BatchNorm2d(planes) if norm == "batch" else Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.activation = Swish()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.activation(self.bn1(x))))
        out = self.conv2(self.activation(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, width_factor, in_channels, norm, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-ResNet depth should be 6n+4"
        n = (depth - 4) / 6
        k = width_factor

        print("Model: Wide-ResNet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(in_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, norm, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, norm, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, norm, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9) if norm == "batch" else Identity()
        self.last_dim = nStages[3]
        self.activation = Swish()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, norm, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, norm, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)

        return out
