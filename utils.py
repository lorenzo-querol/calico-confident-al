from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import yaml
from torch.nn.modules.loss import _Loss


class Hamiltonian(_Loss):
    def __init__(self, layer, reg_cof=1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        y = self.layer(x)
        H = torch.sum(y * p)
        return H


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as file:
        return yaml.safe_load(file)


def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_classification(f, dload, set_name, epoch, args=None, print=None):
    corrects, losses = [], []

    for x, y in dload:
        x, y = x.to(args.device), y.to(args.device)
        logits = f.module.classify(x)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y).detach().cpu().numpy()
        correct = (logits.max(1)[1] == y).float().cpu().numpy()

        losses.extend(loss)
        corrects.extend(correct)

    loss = np.mean(losses)

    return correct, loss
