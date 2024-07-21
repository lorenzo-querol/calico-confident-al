import argparse

import torch
import yaml
from torch.nn.modules.loss import _Loss
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix


class Hamiltonian(_Loss):
    def __init__(self, layer, reg_cof=1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        y = self.layer(x)
        H = torch.sum(y * p)
        return H


def initialize(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_class_dist(labeled_labels, labeled_indices, classes, log_dir):
    labels, counts = np.unique(labeled_labels, return_counts=True)
    distribution_dict = {}
    distribution_dict["num_labeled"] = [len(labeled_indices)]

    for label, count in zip(labels, counts):
        distribution_dict[label] = [count]

    print("Class Distribution:")
    for key, value in distribution_dict.items():
        if key == "num_labeled":
            continue
        print(f"Class {key}: {value[0]}")

    distribution_df = pd.DataFrame(distribution_dict)
    distribution_df.columns = ["num_labeled"] + classes

    if os.path.exists(f"{log_dir}/class_dist.csv"):
        distribution_df.to_csv(f"{log_dir}/class_dist.csv", mode="a", header=False, index=False)
    else:
        distribution_df.to_csv(f"{log_dir}/class_dist.csv", mode="w", header=True, index=False)


# def log_acc_per_class(y_true, y_pred, n_classes, classes, labeled_indices, log_dir):
#     cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
#     accuracies = cm.diagonal() / cm.sum(axis=1)
#     accuracies_df = pd.DataFrame(accuracies).T

#     num_labeled_df = pd.DataFrame([len(labeled_indices)], columns=["num_labeled"])
#     accuracies_df = pd.concat([num_labeled_df, accuracies_df], axis=1)

#     accuracies_df.columns = ["num_labeled"] + classes

#     if os.path.exists(f"{log_dir}/acc_per_class.csv"):
#         accuracies_df.to_csv(f"{log_dir}/acc_per_class.csv", mode="a", header=False, index=False)
#     else:
#         accuracies_df.to_csv(f"{log_dir}/acc_per_class.csv", mode="w", header=True, index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments without default values
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
