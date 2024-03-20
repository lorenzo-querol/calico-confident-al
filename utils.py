import argparse

import torch
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


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments without default values
    parser.add_argument("--model", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--norm", type=str, choices=["none", "batch"])
    parser.add_argument("--decay_epochs", nargs="+", type=int)
    parser.add_argument("--px", type=float)
    parser.add_argument("--pyx", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--in_steps", type=int)
    parser.add_argument("--query_size", type=int)
    parser.add_argument("--sample_method", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--temp_scale", action="store_true")

    # Arguments with default values
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--decay_rate", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clf_only", type=bool, default=False)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=0.05)
    parser.add_argument("--sgld_lr", type=float, default=0.0)
    parser.add_argument("--sgld_std", type=int, default=0)
    parser.add_argument("--pyld_lr", type=float, default=0.2)
    parser.add_argument("--eps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--labels_per_class", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--test_dir", type=str, default="./test_results")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--ckpt_type", type=str, default="last")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--sample_every_n_epochs", type=int, default=10)

    args = parser.parse_args()

    return args


def write_to_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)


def create_log_dir(log_dir):
    counter = 0

    if os.path.exists(log_dir):
        counter += 1

        while os.path.exists(f"{log_dir}-{counter}"):
            counter += 1

        log_dir = f"{log_dir}-{counter}"

    os.makedirs(log_dir, exist_ok=True)

    return log_dir
