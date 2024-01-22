# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import timedelta

import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
from accelerate import Accelerator
from accelerate.utils import set_seed
from netcal.metrics import ECE
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import get_model_and_buffer, get_optimizer
from utils import Hamiltonian, get_experiment_name, parse_args

limit_dict = {
    "cifar10": 40000,
    "cifar100": 40000,
    "svhn": 40000,
    "bloodmnist": 4000,
    "dermamnist": 4000,
    "pneumoniamnist": 4000,
    "organsmnist": 4000,
}

equal_dict = {
    "bloodmnist": 4096,
    "organsmnist": 4400,
}


def main(config):
    accelerator = Accelerator(log_with="wandb" if config["enable_tracking"] else None)
    datamodule = DataModule(accelerator=accelerator, **config)
    datamodule.prepare_data()

    (
        dload_train,
        dload_train_labeled,
        dload_train_unlabeled,
        dload_valid,
        train_labeled_inds,
        train_unlabeled_inds,
    ) = datamodule.get_data(sampling_method="random", init_size=config["query_size"], accelerator=accelerator)

    class_dist = datamodule.get_class_distribution()
    print(class_dist)

    (
        dload_train,
        dload_train_labeled,
        dload_train_unlabeled,
        dload_valid,
    ) = accelerator.prepare(dload_train, dload_train_labeled, dload_train_unlabeled, dload_valid)


if __name__ == "__main__":
    t.backends.cudnn.enabled = True
    t.backends.cudnn.deterministic = True

    config = vars(parse_args())

    """Scale batch size by number of GPUs for reproducibility"""
    config.update({"p_x_weight": 1.0 if config["calibrated"] else 0.0})
    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})
    config.update({"experiment_name": f'{config["dataset"]}_epoch_{config["n_epochs"]}_{config["optimizer"]}'})

    set_seed(config["seed"])

    main(config)
