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

import argparse
import os
import time
from calendar import c
from math import exp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torchvision as tv
from accelerate import Accelerator
from accelerate.utils import set_seed
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataModule import DataModule
from ExpUtils import *
from models.JEM import get_model_and_buffer
from utils import Hamiltonian, load_config, parse_args

conditionals = []


def init_random(datamodule, bs):
    global conditionals

    n_classes = datamodule.n_classes

    n_channels = datamodule.img_shape[0]
    img_shape = datamodule.img_shape
    img_size = datamodule.img_shape[1]

    new = t.zeros(bs, n_channels, img_size, img_size)

    for i in range(bs):
        index = np.random.randint(n_classes)
        dist = conditionals[index]
        new[i] = dist.sample().view(img_shape)

    return t.clamp(new, -1, 1).cpu()


def init_from_centers(
    device, datamodule: DataModule, buffer_size: int, load_path: str = None, **config
):
    global conditionals

    if load_path is not None:
        return t.load(load_path)["replay_buffer"]

    dataset = datamodule.dataset
    n_classes = datamodule.n_classes
    img_shape = datamodule.img_shape
    bs = buffer_size

    centers = t.load(f"weights/{dataset}_mean.pt")
    covs = t.load(f"weights/{dataset}_cov.pt")

    buffer = []
    for i in range(n_classes):
        mean = centers[i].to(device)
        cov = covs[i].to(device)
        dist = MultivariateNormal(
            mean,
            covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(img_shape))).to(device),
        )
        buffer.append(
            dist.sample((bs // n_classes,)).view((bs // n_classes,) + img_shape).cpu()
        )
        conditionals.append(dist)

    return t.clamp(t.cat(buffer), -1, 1)


def sample_p_0(replay_buffer, datamodule, bs, reinit_freq, y=None, **config):
    if len(replay_buffer) == 0:
        return init_random(datamodule, bs), []

    buffer_size = (
        len(replay_buffer) if y is None else len(replay_buffer) // datamodule.n_classes
    )
    inds = t.randint(0, buffer_size, (bs,))

    # If conditional, convert inds to class-conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds

    buffer_samples = replay_buffer[inds]
    random_samples = init_random(datamodule, bs)
    choose_random = (t.rand(bs) < reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples

    return samples.to("cuda"), inds


def sample_q(
    f,
    accelerator,
    datamodule,
    replay_buffer,
    batch_size,
    n_steps,
    in_steps,
    sgld_std,
    sgld_lr,
    pyld_lr,
    eps,
    y=None,
    save=True,
    **config,
):
    bs = batch_size

    init_sample, buffer_inds = sample_p_0(
        replay_buffer=replay_buffer, datamodule=datamodule, bs=bs, y=y, **config
    )
    x_k = t.autograd.Variable(init_sample, requires_grad=True)

    if in_steps > 0:
        Hamiltonian_func = Hamiltonian(accelerator.unwrap_model(f).f.layer_one)

    if pyld_lr <= 0:
        in_steps = 0

    for it in range(n_steps):
        energies = f(x_k)
        e_x = energies.sum()
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]

        if in_steps > 0:
            p = 1.0 * accelerator.unwrap_model(f).f.layer_one_out.grad
            p = p.detach()

        tmp_inp = x_k.data
        tmp_inp.requires_grad_()
        if sgld_lr > 0:
            tmp_inp = x_k + t.clamp(eta, -eps, eps) * sgld_lr
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        for i in range(in_steps):
            H = Hamiltonian_func(tmp_inp, p)

            eta_grad = t.autograd.grad(
                H, [tmp_inp], only_inputs=True, retain_graph=True
            )[0]
            eta_step = t.clamp(eta_grad, -eps, eps) * pyld_lr

            tmp_inp.data = tmp_inp.data + eta_step
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        x_k.data = tmp_inp.data

        if sgld_std > 0.0:
            x_k.data += sgld_std * t.randn_like(x_k)

    if in_steps > 0:
        loss = -1.0 * Hamiltonian_func(x_k.data, p)
        loss.backward()

    final_samples = x_k.detach()

    if len(replay_buffer) > 0 and save:
        replay_buffer[buffer_inds] = final_samples.cpu()

    return final_samples


def category_mean(dload_train, datamodule):
    dataset = datamodule.dataset
    img_shape = datamodule.img_shape
    n_classes = datamodule.n_classes

    centers = t.zeros([n_classes, int(np.prod(img_shape))])
    covs = t.zeros([n_classes, int(np.prod(img_shape)), int(np.prod(img_shape))])

    im_test, targ_test = [], []
    for im, targ in dload_train:
        im_test.append(im)
        targ_test.append(targ)
    im_test, targ_test = t.cat(im_test), t.cat(targ_test)

    for i in range(n_classes):
        if datamodule.dataset in ["cifar10", "cifar100", "svhn"]:
            mask = targ_test == i
        else:
            mask = (targ_test == i).squeeze(1)
        imc = im_test[mask]
        imc = imc.view(len(imc), -1)
        mean = imc.mean(dim=0)
        sub = imc - mean.unsqueeze(dim=0)
        cov = sub.t() @ sub / len(imc)
        centers[i] = mean
        covs[i] = cov

    if not os.path.exists("weights"):
        os.makedirs("weights")

    if datamodule.accelerator.is_main_process:
        t.save(centers, f"weights/{dataset}_mean.pt")
        t.save(covs, f"weights/{dataset}_cov.pt")


def test_model(
    f: nn.Module,
    accelerator: Accelerator,
    datamodule: DataModule,
    test_dir: str,
    num_labeled: int = None,
    **config,
):
    dload_test = datamodule.get_test_data()

    all_corrects, all_losses = [], []
    all_confs, all_gts = [], []
    test_loss, test_acc = np.inf, 0.0

    correct_per_class = {label: 0 for label in datamodule.classnames}
    total_per_class = {label: 0 for label in datamodule.classnames}

    for i, (inputs, labels) in enumerate(
        tqdm(dload_test, desc="Testing", disable=not accelerator.is_main_process)
    ):
        labels = labels.squeeze().long()

        with t.no_grad():
            logits = accelerator.unwrap_model(f).classify(inputs)

        loss, correct, confs, targets = accelerator.gather_for_metrics(
            (
                t.nn.functional.cross_entropy(logits, labels),
                (logits.max(1)[1] == labels).float(),
                t.nn.functional.softmax(logits, dim=1),
                labels,
            )
        )

        all_gts.extend(targets)
        all_confs.extend(confs)
        all_losses.extend(loss)
        all_corrects.extend(correct)

        for i, class_name in enumerate(datamodule.classnames):
            correct_per_class[class_name] += t.sum(
                (correct == 1) & (targets == i)
            ).item()
            total_per_class[class_name] += t.sum(targets == i).item()

    test_loss = np.mean([loss.item() for loss in all_losses])
    test_acc = np.mean([correct.item() for correct in all_corrects])

    accuracy_per_class = {
        label: correct / total if total > 0 else 0
        for label, (correct, total) in zip(
            datamodule.classnames,
            zip(correct_per_class.values(), total_per_class.values()),
        )
    }

    all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape(
        (-1, datamodule.n_classes)
    )
    all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

    ece, diagram = ECE(10), ReliabilityDiagram(10)
    calibration_score = ece.measure(all_confs, all_gts)
    pl = diagram.plot(all_confs, all_gts)

    test_metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_ece": calibration_score,
    }
    test_metrics = pd.DataFrame(test_metrics, index=[0])

    accuracy_per_class = pd.DataFrame(accuracy_per_class, index=[0])

    class_distribution = datamodule.get_class_distribution()
    class_distribution = pd.DataFrame(
        class_distribution, columns=["Class", "Num Samples"]
    )

    if accelerator.is_main_process:
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)

        pl.savefig(f"{test_dir}/reliability_diagram.png")
        plt.close()
        test_metrics.to_csv(f"{test_dir}/test_metrics.csv", index=False)
        accuracy_per_class.to_csv(f"{test_dir}/accuracy_per_class.csv", index=False)
        class_distribution.to_csv(f"{test_dir}/class_distribution.csv", index=False)

    accelerator.print(
        f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | ECE: {calibration_score:.4f}"
    )
    accelerator.log(
        {
            "num_labeled": num_labeled,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_ece": calibration_score,
        }
    )


def get_optimizer(
    accelerator: Accelerator, f: nn.Module, load_path: str = None, **config
):
    """Initialize optimizer"""
    params = f.class_output.parameters() if config["clf_only"] else f.parameters()
    if config["optimizer"] == "adam":
        optim = t.optim.Adam(
            params,
            config["lr"],
            betas=(0.9, 0.999),
            weight_decay=config["weight_decay"],
        )
    else:
        optim = t.optim.SGD(
            params, config["lr"], momentum=0.9, weight_decay=config["weight_decay"]
        )

    if load_path is not None:
        optim.load_state_dict(t.load(load_path)["optimizer_state_dict"])

    if t.cuda.device_count() > 1:
        optim = accelerator.prepare(optim)

    return optim


def get_experiment_name(**config):
    return (
        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{config['dataset']}"
        if config["experiment_name"] is None
        else config["experiment_name"]
    )


def init_logger(
    experiment_name: str,
    experiment_type: str,
    log_dir: str,
    num_labeled: int = None,
    is_test: bool = False,
    seed: int = None,
):
    if is_test:
        run_name = f"test_{experiment_type}_seed_{seed}"
    else:
        run_name = f"active" if experiment_type == "active" else f"baseline"

    logger_kwargs = {
        "group": experiment_name,
        "name": run_name,
    }

    dir_name = (
        f"active_{num_labeled}"
        if experiment_type == "active"
        else f"baseline_{num_labeled}"
    )

    if is_test:
        ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints")
        samples_dir = os.path.join(log_dir, experiment_name, "samples")
        test_dir = os.path.join(log_dir, experiment_name, "test")
    else:
        ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints", dir_name)
        samples_dir = os.path.join(log_dir, experiment_name, "samples", dir_name)
        test_dir = os.path.join(log_dir, experiment_name, "test", dir_name)

    return logger_kwargs, (ckpt_dir, samples_dir, test_dir)


def get_ckpts(ckpt_dir, get_best: bool = True):
    ckpts = list(Path(ckpt_dir).rglob("*"))
    ckpts = [ckpt for ckpt in ckpts if not ckpt.is_dir()]

    if get_best:
        ckpts = [ckpt for ckpt in ckpts if "last" not in ckpt.name]

    ckpts = [str(ckpt) for ckpt in ckpts]

    ckpt_dicts = []
    for ckpt in ckpts:
        path = ckpt.split("/")
        experiment_type, num_labeled = path[-2].split("_")

        ckpt_dicts.append(
            {
                "experiment_type": experiment_type,
                "path": ckpt,
                "num_labeled": num_labeled,
            }
        )

    ckpt_dicts = sorted(ckpt_dicts, key=lambda x: int(x["num_labeled"]))

    return ckpt_dicts


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
    ) = datamodule.get_data(
        sampling_method="random",
        init_size=config["query_size"],
    )

    experiment_name = get_experiment_name(**config)

    logger_kwargs, dirs = init_logger(
        experiment_name=experiment_name,
        experiment_type=config["experiment_type"],
        log_dir=config["log_dir"],
        is_test=True,
        seed=config["seed"],
    )

    ckpt_dir, _, test_dir = dirs
    ckpt_dict_list = get_ckpts(ckpt_dir)

    for ckpt_dict in ckpt_dict_list:
        logger_kwargs, dirs = init_logger(
            experiment_name=experiment_name,
            experiment_type=ckpt_dict["experiment_type"],
            log_dir=config["log_dir"],
            is_test=True,
            seed=config["seed"],
        )

        ckpt_dir, _, test_dir = dirs

        if config["enable_tracking"]:
            accelerator.init_trackers(
                project_name="JEM",
                config=config,
                init_kwargs={
                    "wandb": {
                        "tags": [f'seed_{config["seed"]}', "test"],
                        **logger_kwargs,
                    }
                },
            )

        """Load the best checkpoint"""
        accelerator.print(f"Loading best checkpoint from {ckpt_dict['path']}.")
        f, _ = get_model_and_buffer(
            accelerator=accelerator,
            datamodule=datamodule,
            load_path=ckpt_dict["path"],
            **config,
        )

        """---TESTING---"""
        test_model(
            f=f,
            accelerator=accelerator,
            datamodule=datamodule,
            test_dir=f"{test_dir}/{ckpt_dict['experiment_type']}_{ckpt_dict['num_labeled']}",
            num_labeled=ckpt_dict["num_labeled"],
            **config,
        )

    if config["enable_tracking"]:
        accelerator.end_training()


if __name__ == "__main__":
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.enabled = True
    t.backends.cudnn.deterministic = True

    args = parse_args()
    config = vars(args)

    """Scale batch size by number of GPUs for reproducibility"""
    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})

    if not config["calibrated"]:
        config["p_x_weight"] = 0.0

    set_seed(config["seed"])

    main(config)
