import argparse
import os
import time
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


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments without default values
    parser.add_argument("--model", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--norm", type=str, choices=["none", "batch"])
    parser.add_argument("--decay_epochs", nargs="+", type=int)
    parser.add_argument("--p_x_weight", type=float)
    parser.add_argument("--p_y_x_weight", type=float)
    parser.add_argument("--l2_weight", type=float)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--in_steps", type=int)
    parser.add_argument("--experiment_type", type=str)
    parser.add_argument("--query_size", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--calibrated", action="store_true")

    # Arguments with default values
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--decay_rate", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clf_only", type=bool, default=False)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=0.05)
    parser.add_argument("--sgld_lr", type=float, default=0.0)
    parser.add_argument("--sgld_std", type=int, default=0)
    parser.add_argument("--pyld_lr", type=float, default=0.2)
    parser.add_argument("--eps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--labels_per_class", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--ckpt_every_n_epochs", type=int, default=None)
    parser.add_argument("--sample_every_n_epochs", type=int, default=10)
    parser.add_argument("--enable_tracking", action="store_true")

    parser.add_argument("--run_once", action="store_true")

    args = parser.parse_args()

    return args


def get_experiment_name(dataset: str, experiment_name: str, **config):
    return f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{dataset}" if experiment_name is None else experiment_name


def get_logger_kwargs(experiment_name: str, experiment_type: str, seed: int, **config):
    """
    Returns a dictionary of keyword arguments to pass to the logger. The logger will log the experiment to the
    `experiment_name` group and name the run `test_{experiment_type}_seed_{seed}`.

    Params:
    - experiment_name: The name of the experiment.
    - experiment_type: The type of experiment (e.g. "jvq").
    - seed: The seed used for the experiment.
    - config: The configuration dictionary.

    Return:
    - A dictionary containing group and name of the project.
    """
    run_name = f"test_{experiment_type}_seed_{seed}"

    logger_kwargs = {
        "group": experiment_name,
        "name": run_name,
    }

    return logger_kwargs


def get_directories(log_dir: str, experiment_name: str, seed: int, **config):
    ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints")
    samples_dir = os.path.join(log_dir, experiment_name, "samples")
    test_dir = os.path.join(log_dir, experiment_name, "test", f"seed_{seed}")

    return ckpt_dir, samples_dir, test_dir
