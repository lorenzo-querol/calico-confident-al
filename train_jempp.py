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
from accelerate.utils import LoggerType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torchvision as tv
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from wandb.util import generate_id

import wandb
from ExpUtils import *
from models.jem_models import get_model_and_buffer
from utils import (
    DataModule,
    Hamiltonian,
    eval_classification,
    load_config,
    plot,
)

t.set_num_threads(2)
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
inner_his = []
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
    global inner_his
    inner_his = []

    bs = batch_size if y is None else y.size(0)

    init_sample, buffer_inds = sample_p_0(replay_buffer=replay_buffer, datamodule=datamodule, bs=bs, y=y, **config)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)

    if in_steps > 0:
        Hamiltonian_func = Hamiltonian(accelerator.unwrap_model(f).f.layer_one)

    if pyld_lr <= 0:
        in_steps = 0

    for it in range(n_steps):
        energies = f(x_k, y=y)
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


def init_from_centers(accelerator, datamodule, buffer_size, **config):
    global conditionals

    device = accelerator.device
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


def init_logger(experiment_name, experiment_type, log_dir, iter_num=None, num_labeled=None, **kwargs):
    al_run_name = f"al_iter_{iter_num}"
    baseline_run_name = f"baseline_{num_labeled}"

    logger_kwargs = {"group": experiment_name, "name": "active" if experiment_type == "active" else "baseline"}
    ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints", al_run_name if experiment_type == "active" else baseline_run_name)
    samples_dir = os.path.join(log_dir, experiment_name, "samples", al_run_name)
    test_dir = os.path.join(log_dir, experiment_name, "test", al_run_name if experiment_type == "active" else baseline_run_name)

    return logger_kwargs, ckpt_dir, samples_dir, test_dir


def train_model(
    f,
    optim,
    accelerator,
    datamodule,
    dload_train,
    dload_train_labeled,
    dload_valid,
    train_labeled_inds,
    train_unlabeled_inds,
    replay_buffer,
    exp_name,
    iter_num,
    **config,
):
    logger_kwargs, ckpt_dir, samples_dir, test_dir = init_logger(
        experiment_name=exp_name,
        experiment_type=config["experiment_type"],
        log_dir=config["log_dir"],
        iter_num=iter_num + 1,
        num_labeled=len(train_labeled_inds),
    )

    if config["enable_tracking"]:
        accelerator.init_trackers(project_name="JEM", config=config, init_kwargs={"wandb": logger_kwargs})

    cur_iter = 0
    new_lr = config["lr"]
    best_val_loss = np.inf
    best_ckpt_path = None

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

            epoch_loss += L
            epoch_acc += acc.item()
            epoch_loss_p_x += loss_p_x.item() if config["p_x_weight"] > 0 else 0.0
            epoch_loss_p_y_x += loss_p_y_x.item()

            """Take gradient step"""
            optim.zero_grad()
            accelerator.backward(L)
            optim.step()
            cur_iter += 1

        """---VALIDATION---"""

        f.eval()
        corrects, losses = [], []
        val_loss, val_acc = 0.0, 0.0
        for x, y in dload_valid:
            y = y.squeeze().long()

            with t.no_grad():
                logits = accelerator.unwrap_model(f).classify(x)

            loss = nn.functional.cross_entropy(logits, y, reduction="none")
            correct = (logits.max(1)[1] == y).float()

            losses.extend(accelerator.gather(loss.reshape(-1)))
            corrects.extend(accelerator.gather(correct.reshape(-1)))

        val_loss = np.mean([loss.item() for loss in losses])
        val_acc = np.mean([correct.item() for correct in corrects])

        """---LOGGING AND CHECKPOINTING---"""

        if epoch % config["sample_every_n_epochs"] == 0 and config["p_x_weight"] > 0:
            with accelerator.no_sync(f):
                x_q = sample_q(f, accelerator, datamodule, replay_buffer, **config)

            if accelerator.is_main_process:
                image = tv.utils.make_grid(x_q, normalize=True, nrow=8, value_range=(-1, 1))

                if not os.path.exists(samples_dir):
                    os.makedirs(samples_dir, exist_ok=True)

                tv.utils.save_image(image, f"{samples_dir}/x_q-epoch={epoch}.png")

                if config["enable_tracking"]:
                    accelerator.log(
                        {"sampled_images": wandb.Image(image, caption="Sampled Images")}, step=epoch + (config["n_epochs"] * (iter_num + 1))
                    )

        if config["ckpt_every_n_epochs"] and epoch % config["ckpt_every_n_epochs"] == 0:
            ckpt_dict = {
                "model_state_dict": accelerator.unwrap_model(f).state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "replay_buffer": replay_buffer,
            }

            if accelerator.is_main_process:
                accelerator.save(ckpt_dict, f"{ckpt_dir}/epoch={epoch}.ckpt")

        """Check if current valid loss is the best"""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.print(f"Best Valid Loss: {best_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

            ckpt_dict = {
                "model_state_dict": accelerator.unwrap_model(f).state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "replay_buffer": replay_buffer,
            }

            if accelerator.is_main_process:
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)

                best_ckpt_path = f"{ckpt_dir}/epoch={epoch}-val_loss={val_loss:.4f}.ckpt"
                os.makedirs(ckpt_dir, exist_ok=True)

                accelerator.save(ckpt_dict, f"{ckpt_dir}/epoch={epoch}-val_loss={val_loss:.4f}.ckpt")

        epoch_loss /= len(dload_train)
        epoch_acc /= len(dload_train)
        epoch_loss_p_x /= len(dload_train)
        epoch_loss_p_y_x /= len(dload_train)

        metric_print = f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Loss P(x): {epoch_loss_p_x:.4f}, Loss P(y|x): {epoch_loss_p_y_x:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        accelerator.print(metric_print)

        if config["enable_tracking"]:
            values = {
                "epoch": epoch + (config["n_epochs"] * (iter_num + 1)),
                "loss": epoch_loss,
                "loss_p_x": epoch_loss_p_x if config["p_x_weight"] > 0 else 0.0,
                "loss_p_y_x": epoch_loss_p_y_x,
                "acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            accelerator.log(values)

    """Save the last checkpoint"""
    if accelerator.is_main_process:
        ckpt_dict = {
            "model_state_dict": accelerator.unwrap_model(f).state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "replay_buffer": replay_buffer,
        }
        accelerator.save(ckpt_dict, f"{ckpt_dir}/last.ckpt")

    return f, test_dir


def test_model(datamodule, accelerator, f, config, test_dir):
    dload_test = datamodule.get_test_data()
    dload_test = accelerator.prepare(dload_test)

    corrects, losses = [], []
    confs, gts = [], []
    test_loss, test_acc = 0.0, 0.0

    correct_per_class = {label: 0 for label in datamodule.classnames}
    total_per_class = {label: 0 for label in datamodule.classnames}

    for x, y in dload_test:
        y = y.squeeze().long()

        with t.no_grad():
            logits = accelerator.unwrap_model(f).classify(x)

        loss = t.nn.functional.cross_entropy(logits, y, reduction="none")
        correct = (logits.max(1)[1] == y).float()

        for i in range(datamodule.n_classes):
            label = datamodule.classnames[i]
            correct_per_class[label] += t.sum((correct == 1) & (y == i)).item()
            total_per_class[label] += t.sum(y == i).item()

        gts.extend(accelerator.gather(y.reshape(-1)))
        confs.extend(accelerator.gather(t.nn.functional.softmax(logits, 1)))

        losses.extend(loss.reshape(-1))
        corrects.extend(correct.reshape(-1))

    test_loss = np.mean([loss.item() for loss in losses])
    test_acc = np.mean([correct.item() for correct in corrects])

    accuracy_per_class = {
        label: correct / total if total > 0 else 0
        for label, (correct, total) in zip(datamodule.classnames, zip(correct_per_class.values(), total_per_class.values()))
    }

    confs = np.array([conf.cpu().numpy() for conf in confs]).reshape((-1, datamodule.n_classes))
    gts = np.array([gt.cpu().numpy() for gt in gts])

    ece, diagram = ECE(10), ReliabilityDiagram(10)
    calibration_score = ece.measure(confs, gts)
    pl = diagram.plot(confs, gts)

    test_metrics = {"test_loss": test_loss, "test_acc": test_acc, "test_ece": calibration_score}
    test_metrics = pd.DataFrame(test_metrics, index=[0])
    accuracy_per_class = pd.DataFrame(accuracy_per_class, index=[0])

    if accelerator.is_main_process:
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)

        pl.savefig(f"{test_dir}/reliability_diagram.png")
        test_metrics.to_csv(f"{test_dir}/test_metrics.csv", index=False)
        accuracy_per_class.to_csv(f"{test_dir}/accuracy_per_class.csv", index=False)

    accelerator.print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | ECE: {calibration_score:.4f}")
    # accelerator.log({"test_loss": test_loss, "test_acc": test_acc, "ece": calibration_score})

    # if config["enable_tracking"]:
    # accelerator.log({"reliability_diagram": wandb.Image(pl, caption="Reliability Diagram")})
    # accelerator.log({"accuracy_per_class": wandb.Table(dataframe=accuracy_per_class)})


def get_optimizer(f, config):
    """Initialize optimizer"""
    params = f.class_output.parameters() if config["clf_only"] else f.parameters()
    if config["optimizer"] == "adam":
        optim = t.optim.Adam(params, lr=config["lr"], betas=(0.9, 0.999), weight_decay=config["weight_decay"])
    else:
        optim = t.optim.SGD(params, lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])

    return optim


def get_experiment_name(config):
    if config["experiment_name"] is None:
        experiment_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{config['dataset']}_{config['group_id']}"
    else:
        experiment_name = config["experiment_name"]

    return experiment_name


def main(config):
    accelerator = Accelerator(log_with=["wandb"] if config["enable_tracking"] else None)
    datamodule = DataModule(accelerator=accelerator, **config)
    experiment_name = get_experiment_name(config=config)

    """For informative initialization"""
    if not os.path.isfile(f"weights/{datamodule.dataset}_cov.pt") and accelerator.is_main_process:
        category_mean(dload_train=dload_train, datamodule=datamodule)

    if config["experiment_type"] == "active":
        dload_train, dload_train_labeled, dload_train_unlabeled, dload_valid, train_labeled_inds, train_unlabeled_inds = datamodule.get_data()
        f, replay_buffer = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, **config)
        replay_buffer = init_from_centers(accelerator=accelerator, datamodule=datamodule, **config)
        optim = get_optimizer(f=f, config=config)

        """Prepare model, optimizer for DDP"""
        f, optim = accelerator.prepare(f, optim)

        for i in range(config["n_al_iters"]):
            accelerator.print(f"Active Learning Iteration {i + 1}")

            """---TRAINING---"""

            f, test_dir = train_model(
                f=f,
                optim=optim,
                accelerator=accelerator,
                datamodule=datamodule,
                dload_train=dload_train,
                dload_train_labeled=dload_train_labeled,
                dload_valid=dload_valid,
                train_labeled_inds=train_labeled_inds,
                train_unlabeled_inds=train_unlabeled_inds,
                replay_buffer=replay_buffer,
                exp_name=experiment_name,
                iter_num=i,
                **config,
            )
            accelerator.wait_for_everyone()

            """---TESTING---"""

            test_model(datamodule=datamodule, accelerator=accelerator, f=f, config=config, test_dir=test_dir)
            accelerator.wait_for_everyone()

            """---ACTIVE LEARNING STEP---"""

            if accelerator.is_main_process:
                counts = datamodule.get_class_dist()
                counts = pd.DataFrame(counts, columns=["Class", "Num Samples"])
                counts.to_csv(f"{test_dir}/class_dist.csv", index=False)

            # if config["enable_tracking"]:
            #     accelerator.log({"class_distribution": wandb.Table(data=datamodule.get_class_dist(), columns=["Class", "Num Samples"])})

            inds_to_fix = datamodule.query_samples(f, dload_train_unlabeled, train_unlabeled_inds, datamodule.n_classes, config["query_size"])

            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(train_labeled_inds, train_unlabeled_inds, inds_to_fix, start_iter=False)

    if config["experiment_type"] == "baseline":
        for i in range(config["n_al_iters"]):
            accelerator.print(f"Baseline Experiment with {config['labels_per_class'] * (i + 1)} labels per class")

            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(override_labels_per_class=config["labels_per_class"] * (i + 1))

            f, replay_buffer = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, **config)
            replay_buffer = init_from_centers(accelerator=accelerator, datamodule=datamodule, **config)
            optim = get_optimizer(f=f, config=config)

            """Prepare model, optimizer for DDP"""
            f, optim = accelerator.prepare(f, optim)

            """---TRAINING---"""

            f, test_dir = train_model(
                f=f,
                optim=optim,
                accelerator=accelerator,
                datamodule=datamodule,
                dload_train=dload_train,
                dload_train_labeled=dload_train_labeled,
                dload_valid=dload_valid,
                train_labeled_inds=train_labeled_inds,
                train_unlabeled_inds=train_unlabeled_inds,
                replay_buffer=replay_buffer,
                exp_name=experiment_name,
                iter_num=i,
                **config,
            )
            accelerator.wait_for_everyone()

            """---TESTING---"""

            test_model(datamodule=datamodule, accelerator=accelerator, f=f, config=config, test_dir=test_dir)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                counts = datamodule.get_class_dist()
                counts = pd.DataFrame(counts, columns=["Class", "Num Samples"])
                counts.to_csv(f"{test_dir}/class_dist.csv", index=False)

    if config["enable_tracking"]:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Active Learning with JEM++")

    parser.add_argument("--model_config", type=str, default="configs/jempp_hparams.yml", help="Path to the config file.")
    parser.add_argument("--dataset_config", type=str, default="configs/cifar10.yml", help="Path to the config file.")
    parser.add_argument("--logging_config", type=str, default="configs/logging.yml", help="Path to the config file.")
    args = parser.parse_args()

    model_config = load_config(Path(args.model_config))
    dataset_config = load_config(Path(args.dataset_config))
    logging_config = load_config(Path(args.logging_config))
    config = {**model_config, **dataset_config, **logging_config}

    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})

    if not config["calibrated"]:
        config["p_x_weight"] = 0.0

    if config["experiment_name"] is None:
        config["group_id"] = generate_id()

    set_seed(config["seed"])

    main(config)
