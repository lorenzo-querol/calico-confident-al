from argparse import ArgumentParser

import torch.nn as nn


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def get_args():
    parser = ArgumentParser()

    # Data
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--sigma", type=float, default=3e-2)
    parser.add_argument("--dataset", type=str)  # No defaults for these

    # Model
    # parser.add_argument("--model", type=str, default="wrn", choices=["wrn", "cnn"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--decay_epochs", type=int, default=[60, 120, 180], nargs="+")
    parser.add_argument("--decay_rate", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--norm", type=str, default="batch", choices=["none", "batch"])
    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=0.1)

    # Loss weighting
    parser.add_argument("--px", type=float, default=1.0)
    parser.add_argument("--pyx", type=float, default=1.0)

    # JEM
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=0.0)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=0.05)

    # SAM Optimizer
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--rho", type=float, default=2.0)

    # Logging
    parser.add_argument("--enable_log", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--sample_every_n_epochs", type=int, default=10)
    parser.add_argument("--exp_name", type=str)  # No defaults for these

    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()
