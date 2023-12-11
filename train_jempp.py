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
from utils import Hamiltonian, load_config

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


def init_from_centers(device, datamodule: DataModule, buffer_size: int, load_path: str = None, **config):
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
        dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(img_shape))).to(device))
        buffer.append(dist.sample((bs // n_classes,)).view((bs // n_classes,) + img_shape).cpu())
        conditionals.append(dist)

    return t.clamp(t.cat(buffer), -1, 1)


def sample_p_0(replay_buffer, datamodule, bs, reinit_freq, y=None, **config):
    if len(replay_buffer) == 0:
        return init_random(datamodule, bs), []

    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // datamodule.n_classes
    inds = t.randint(0, buffer_size, (bs,))

    # If conditional, convert inds to class-conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds

    buffer_samples = replay_buffer[inds]
    random_samples = init_random(datamodule, bs)
    choose_random = (t.rand(bs) < reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples

    return samples.to("cuda"), inds


def sample_q(f, accelerator, datamodule, replay_buffer, batch_size, n_steps, in_steps, sgld_std, sgld_lr, pyld_lr, eps, y=None, save=True, **config):
    bs = batch_size

    init_sample, buffer_inds = sample_p_0(replay_buffer=replay_buffer, datamodule=datamodule, bs=bs, y=y, **config)
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

            eta_grad = t.autograd.grad(H, [tmp_inp], only_inputs=True, retain_graph=True)[0]
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
        if datamodule.dataset == "cifar10":
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

    t.save(centers, f"weights/{dataset}_mean.pt")
    t.save(covs, f"weights/{dataset}_cov.pt")


def train_model(
    f: nn.Module,
    optim: t.optim.Optimizer,
    accelerator: Accelerator,
    datamodule: DataModule,
    dload_train: DataLoader,
    dload_train_labeled: DataLoader,
    dload_valid: DataLoader,
    replay_buffer: t.Tensor,
    dirs: tuple[str, str, str],
    iter_num: int = 0,
    **config,
):
    ckpt_dir, samples_dir, _ = dirs

    cur_iter = 0
    new_lr = config["lr"]
    best_val_loss = np.inf
    best_val_acc = 0.0
    best_ckpt_path = None
    reset_decay = 1.0

    for epoch in range(config["n_epochs"]):
        if epoch in config["decay_epochs"]:
            for param_group in optim.param_groups:
                new_lr = param_group["lr"] * config["decay_rate"]
                param_group["lr"] = new_lr
            accelerator.print(f"Decaying LR to {new_lr:.8f}.")

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_loss_p_x = 0.0
        epoch_loss_p_y_x = 0.0
        epoch_loss_l2 = 0.0
        loss_p_x = 0.0
        progress_bar = tqdm(dload_train, desc=(f"Epoch {epoch}"), disable=not accelerator.is_main_process)

        """---TRAINING---"""

        f.train()
        for i, (x_p_d, _) in enumerate(progress_bar):
            """Warmup Learning Rate"""
            if cur_iter <= config["warmup_iters"]:
                lr = config["lr"] * cur_iter / float(config["warmup_iters"])
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(accelerator.device), y_lab.to(accelerator.device).squeeze().long()

            L = 0.0

            """Maximize log P(x)"""
            if config["p_x_weight"] > 0:
                with accelerator.no_sync(f):
                    fp_all = f(x_p_d)
                    fp = fp_all.mean()

                    x_q = sample_q(f, accelerator, datamodule, replay_buffer, **config)
                    fq_all = f(x_q)
                    fq = fq_all.mean()

                    loss_p_x = -(fp - fq)
                    L += config["p_x_weight"] * loss_p_x

            """Maximize log P(y|x)"""
            if config["p_y_x_weight"] > 0:
                logits = accelerator.unwrap_model(f).classify(x_lab)
                loss_p_y_x = nn.functional.cross_entropy(logits, y_lab)
                acc = (logits.max(1)[1] == y_lab).float().mean()
                L += config["p_y_x_weight"] * loss_p_y_x

            if config["l2_weight"] > 0:
                loss_l2 = (fq**2 + fp**2).mean() * config["l2_weight"]
                L += loss_l2

            epoch_loss += L
            epoch_acc += acc.item()
            epoch_loss_p_x += loss_p_x.item() if config["p_x_weight"] > 0 else 0.0
            epoch_loss_l2 += loss_l2.item() if config["l2_weight"] > 0 else 0.0
            epoch_loss_p_y_x += loss_p_y_x.item()

            """Take gradient step"""
            optim.zero_grad()
            accelerator.backward(L)
            optim.step()
            cur_iter += 1

        """---VALIDATION---"""

        f.eval()
        all_corrects, all_losses = [], []
        val_loss, val_acc = np.inf, 0.0
        for inputs, labels in dload_valid:
            labels = labels.squeeze().long()

            with t.no_grad():
                logits = accelerator.unwrap_model(f).classify(inputs)

            losses, corrects = accelerator.gather_for_metrics((t.nn.functional.cross_entropy(logits, labels), (logits.max(1)[1] == labels).float()))

            all_losses.extend(losses)
            all_corrects.extend(corrects)

        val_loss = np.mean([loss.item() for loss in all_losses])
        val_acc = np.mean([correct.item() for correct in all_corrects])

        """---LOGGING AND CHECKPOINTING---"""

        if (epoch + (config["n_epochs"] * iter_num)) % config["sample_every_n_epochs"] == 0 and config["p_x_weight"] > 0:
            with accelerator.no_sync(f):
                x_q = sample_q(f, accelerator, datamodule, replay_buffer, **config)

            image = tv.utils.make_grid(x_q, normalize=True, nrow=8, value_range=(-1, 1))

            if accelerator.is_main_process:
                if not os.path.exists(samples_dir):
                    os.makedirs(samples_dir, exist_ok=True)

                tv.utils.save_image(image, f"{samples_dir}/x_q-epoch={epoch + (config['n_epochs'] * iter_num)}.png")

        if config["ckpt_every_n_epochs"] and (epoch + (config["n_epochs"] * iter_num)) % config["ckpt_every_n_epochs"] == 0:
            ckpt_dict = {
                "model_state_dict": accelerator.unwrap_model(f).state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "replay_buffer": replay_buffer,
            }

            if accelerator.is_main_process:
                accelerator.save(ckpt_dict, f"{ckpt_dir}/epoch={epoch + (config['n_epochs'] * iter_num)}.ckpt")

        """Check if current valid loss is the best"""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            accelerator.print(f"BEST Val Loss: {best_val_loss:.4f} \t Val Accuracy: {val_acc:.4f}")

            if config["enable_tracking"]:
                accelerator.log({"val_loss": best_val_loss, "val_acc": val_acc})
            ckpt_dict = {
                "model_state_dict": accelerator.unwrap_model(f).state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "replay_buffer": replay_buffer,
            }

            if accelerator.is_main_process:
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)

                best_ckpt_path = f"{ckpt_dir}/epoch={epoch + (config['n_epochs'] * iter_num)}-val_loss={val_loss:.4f}.ckpt"
                os.makedirs(ckpt_dir, exist_ok=True)

                accelerator.save(ckpt_dict, best_ckpt_path)

        epoch_loss /= len(dload_train)
        epoch_acc /= len(dload_train)
        epoch_loss_p_x /= len(dload_train)
        epoch_loss_p_y_x /= len(dload_train)
        epoch_loss_l2 /= len(dload_train)

        metric_print = f"Loss: {epoch_loss:.4f} \t Acc: {epoch_acc:.4f} \t Loss P(x): {epoch_loss_p_x:.4f} \t  Loss L2: {epoch_loss_l2:.4f} \t Loss P(y|x): {epoch_loss_p_y_x:.4f} \t Val Loss: {val_loss:.4f} \t Val Acc: {val_acc:.4f}"
        accelerator.print(metric_print)

        if config["enable_tracking"]:
            accelerator.log(
                {
                    "epoch": epoch + (config["n_epochs"] * iter_num),
                    "loss": epoch_loss,
                    "loss_p_x": epoch_loss_p_x if config["p_x_weight"] > 0 else 0.0,
                    "loss_l2": epoch_loss_l2 if config["l2_weight"] > 0 else 0.0,
                    "loss_p_y_x": epoch_loss_p_y_x,
                    "acc": epoch_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

    """Log with respect to number of labeled samples"""
    if config["enable_tracking"]:
        accelerator.log(
            {
                "num_labeled": len(datamodule.train_labeled_indices),
                "loss": epoch_loss,
                "acc": epoch_acc,
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
            }
        )

    """Save the last checkpoint"""
    ckpt_dict = {
        "model_state_dict": accelerator.unwrap_model(f).state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "replay_buffer": replay_buffer,
    }

    last_ckpt_path = f"{ckpt_dir}/last.ckpt"
    if accelerator.is_main_process:
        accelerator.save(ckpt_dict, f"{ckpt_dir}/last.ckpt")

    return f, best_ckpt_path, last_ckpt_path


def test_model(f: nn.Module, accelerator: Accelerator, datamodule: DataModule, dirs: tuple[int, int, int], **config):
    _, _, test_dir = dirs
    dload_test = datamodule.get_test_data()

    all_corrects, all_losses = [], []
    all_confs, all_gts = [], []
    test_loss, test_acc = np.inf, 0.0

    correct_per_class = {label: 0 for label in datamodule.classnames}
    total_per_class = {label: 0 for label in datamodule.classnames}

    for i, (inputs, labels) in enumerate(tqdm(dload_test, desc="Testing", disable=not accelerator.is_main_process)):
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
            correct_per_class[class_name] += t.sum((correct == 1) & (targets == i)).item()
            total_per_class[class_name] += t.sum(targets == i).item()

    test_loss = np.mean([loss.item() for loss in all_losses])
    test_acc = np.mean([correct.item() for correct in all_corrects])

    accuracy_per_class = {
        label: correct / total if total > 0 else 0
        for label, (correct, total) in zip(datamodule.classnames, zip(correct_per_class.values(), total_per_class.values()))
    }

    all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape((-1, datamodule.n_classes))
    all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

    ece, diagram = ECE(10), ReliabilityDiagram(10)
    calibration_score = ece.measure(all_confs, all_gts)
    pl = diagram.plot(all_confs, all_gts)

    test_metrics = {"test_loss": test_loss, "test_acc": test_acc, "test_ece": calibration_score}
    test_metrics = pd.DataFrame(test_metrics, index=[0])

    accuracy_per_class = pd.DataFrame(accuracy_per_class, index=[0])

    class_distribution = datamodule.get_class_distribution()
    class_distribution = pd.DataFrame(class_distribution, columns=["Class", "Num Samples"])

    if accelerator.is_main_process:
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)

        pl.savefig(f"{test_dir}/reliability_diagram.png")
        plt.close()
        test_metrics.to_csv(f"{test_dir}/test_metrics.csv", index=False)
        accuracy_per_class.to_csv(f"{test_dir}/accuracy_per_class.csv", index=False)
        class_distribution.to_csv(f"{test_dir}/class_distribution.csv", index=False)

    accelerator.print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | ECE: {calibration_score:.4f}")
    accelerator.log(
        {
            "num_labeled": len(datamodule.train_labeled_indices),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_ece": calibration_score,
        }
    )


def get_optimizer(accelerator: Accelerator, f: nn.Module, load_path: str = None, **config):
    """Initialize optimizer"""
    params = f.class_output.parameters() if config["clf_only"] else f.parameters()
    if config["optimizer"] == "adam":
        optim = t.optim.Adam(params, config["lr"], betas=(0.9, 0.999), weight_decay=config["weight_decay"])
    else:
        optim = t.optim.SGD(params, config["lr"], momentum=0.9, weight_decay=config["weight_decay"])

    if load_path is not None:
        optim.load_state_dict(t.load(load_path)["optimizer_state_dict"])

    if t.cuda.device_count() > 1:
        optim = accelerator.prepare(optim)

    return optim


def get_experiment_name(**config):
    return f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{config['dataset']}" if config["experiment_name"] is None else config["experiment_name"]


def init_logger(experiment_name, experiment_type, log_dir, num_labeled=None, **config):
    dir_name = f"active_{num_labeled}" if experiment_type == "active" else f"baseline_{num_labeled}"
    run_name = "active" if experiment_type == "active" else f"baseline"

    logger_kwargs = {
        "group": experiment_name,
        "name": run_name,
    }
    ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints", dir_name)
    samples_dir = os.path.join(log_dir, experiment_name, "samples", dir_name)
    test_dir = os.path.join(log_dir, experiment_name, "test", dir_name)

    return logger_kwargs, (ckpt_dir, samples_dir, test_dir)


def main(config):
    accelerator = Accelerator(log_with="wandb" if config["enable_tracking"] else None)
    datamodule = DataModule(accelerator=accelerator, **config)

    experiment_name = get_experiment_name(**config)

    """For informative initialization"""
    if not os.path.isfile(f"weights/{datamodule.dataset}_cov.pt") and accelerator.is_main_process:
        category_mean(dload_train=dload_train, datamodule=datamodule)

    """Randomly initialize the labeled training pool"""
    dload_train, dload_train_labeled, dload_train_unlabeled, dload_valid, train_labeled_inds, train_unlabeled_inds = datamodule.get_data(
        sampling_method="random",
        init_size=config["query_size"],
    )

    f, replay_buffer = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, **config)
    replay_buffer = init_from_centers(device=accelerator.device, datamodule=datamodule, **config)
    optim = get_optimizer(accelerator=accelerator, f=f, **config)

    n_iters = len(datamodule.full_train) // config["query_size"]
    logger_kwargs, dirs = init_logger(experiment_name=experiment_name, experiment_type=config["experiment_type"], log_dir=config["log_dir"])
    init_size = config["query_size"]

    if config["enable_tracking"]:
        accelerator.init_trackers(project_name="JEM", init_kwargs={"wandb": logger_kwargs})

    for i in range(n_iters):
        logger_kwargs, dirs = init_logger(
            experiment_name=experiment_name,
            experiment_type=config["experiment_type"],
            log_dir=config["log_dir"],
            num_labeled=len(train_labeled_inds),
        )

        """---TRAINING---"""
        f, best_ckpt_path, last_ckpt_path = train_model(
            f=f,
            optim=optim,
            accelerator=accelerator,
            datamodule=datamodule,
            dload_train=dload_train,
            dload_train_labeled=dload_train_labeled,
            dload_valid=dload_valid,
            train_labeled_inds=train_labeled_inds,
            replay_buffer=replay_buffer,
            dirs=dirs,
            iter_num=i,
            **config,
        )

        """Load the best checkpoint"""
        accelerator.print(f"Loading best checkpoint from {best_ckpt_path}.")
        f, replay_buffer = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, load_path=best_ckpt_path, **config)

        """---TESTING---"""
        test_model(f=f, accelerator=accelerator, datamodule=datamodule, dirs=dirs, **config)

        if config["experiment_type"] == "active":
            """---ACTIVE LEARNING STEP---"""
            inds_to_fix = datamodule.query_samples(
                f=f,
                dload_train_unlabeled=dload_train_unlabeled,
                train_unlabeled_inds=train_unlabeled_inds,
                n_classes=datamodule.n_classes,
                query_size=config["query_size"],
            )
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(
                train_labeled_indices=train_labeled_inds,
                train_unlabeled_indices=train_unlabeled_inds,
                indices_to_fix=inds_to_fix,
                start_iter=False,
            )
        else:
            """---BASELINE STEP---"""
            init_size += config["query_size"]
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(start_iter=False, sampling_method="random", init_size=init_size)

        """---REINITIALIZE---"""
        f, replay_buffer = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, **config)  # From scratch
        replay_buffer = init_from_centers(device=accelerator.device, datamodule=datamodule, load_path=last_ckpt_path, **config)  # From last ckpt
        optim = get_optimizer(accelerator=accelerator, f=f, **config)  # From scratch

    if config["enable_tracking"]:
        accelerator.end_training()


if __name__ == "__main__":
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.enabled = True
    t.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser("Active Learning with JEM++")
    parser.add_argument("--model_config", type=str, default="configs/jempp_hparams.yml", help="Path to the config file.")
    parser.add_argument("--dataset_config", type=str, default="configs/cifar10.yml", help="Path to the config file.")
    parser.add_argument("--logging_config", type=str, default="configs/logging.yml", help="Path to the config file.")
    args = parser.parse_args()

    model_config = load_config(Path(args.model_config))
    dataset_config = load_config(Path(args.dataset_config))
    logging_config = load_config(Path(args.logging_config))
    config = {**model_config, **dataset_config, **logging_config}

    """Scale batch size by number of GPUs for reproducibility"""
    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})

    if not config["calibrated"]:
        config["p_x_weight"] = 0.0

    set_seed(config["seed"])

    main(config)
