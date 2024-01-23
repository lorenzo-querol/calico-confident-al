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
import wandb

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
        dist = MultivariateNormal(
            mean,
            covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(img_shape))).to(device),
        )
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


def sample_q(
    f, datamodule, replay_buffer, batch_size, n_steps, in_steps, sgld_std, sgld_lr, pyld_lr, eps, y=None, save=True, accelerator=None, **config
):
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
            p = 1.0 * accelerator.unwrap_model(f).f.layer_one_out.grad if accelerator else f.f.layer_one_out.grad
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
        if datamodule.dataset in ["cifar10", "cifar100", "svhn", "mnist"]:
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

    if datamodule.accelerator:
        if datamodule.accelerator.is_main_process:
            t.save(centers, f"weights/{dataset}_mean.pt")
            t.save(covs, f"weights/{dataset}_cov.pt")
    else:
        t.save(centers, f"weights/{dataset}_mean.pt")
        t.save(covs, f"weights/{dataset}_cov.pt")


def train_model(
    f: nn.Module,
    optim: t.optim.Optimizer,
    datamodule: DataModule,
    dload_train: DataLoader,
    dload_train_labeled: DataLoader,
    dload_valid: DataLoader,
    replay_buffer: t.Tensor,
    dirs: tuple[str, str, str],
    accelerator: Accelerator = None,
    iter_num: int = 0,
    **config,
):
    print_fn = accelerator.print if accelerator else print
    log_fn = accelerator.log if accelerator else wandb.log
    device = accelerator.device if accelerator else t.device("cuda")
    ckpt_dir, samples_dir, _ = dirs

    cur_iter = 0
    new_lr = config["lr"]
    best_val_loss, best_val_acc, best_val_ece = np.inf, 0.0, 0.0
    best_ckpt_path = None

    for epoch in range(config["n_epochs"]):
        if epoch in config["decay_epochs"]:
            for param_group in optim.param_groups:
                new_lr = param_group["lr"] * config["decay_rate"]
                param_group["lr"] = new_lr

            print_fn(f"Decaying LR to {new_lr:.8f}.")

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_loss_p_x = 0.0
        epoch_loss_p_y_x = 0.0
        epoch_loss_l2 = 0.0
        loss_p_x = 0.0
        loss_l2 = 0.0

        """---TRAINING---"""

        progress_bar = tqdm(
            dload_train,
            desc=(f"Epoch {epoch+1}/{config['n_epochs']}"),
            disable=not accelerator.is_main_process if accelerator else False,
        )

        f.train()
        for i, (x_p_d, _) in enumerate(progress_bar):
            """Warmup Learning Rate"""
            if cur_iter <= config["warmup_iters"]:
                lr = config["lr"] * cur_iter / float(config["warmup_iters"])
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = (x_lab.to(device), y_lab.to(device).squeeze().long())

            L = 0.0

            """Maximize log P(x)"""
            if config["p_x_weight"] > 0:
                if accelerator is not None:
                    with accelerator.no_sync(f):
                        fp_all = f(x_p_d)
                        fp = fp_all.mean()

                        x_q = sample_q(f, accelerator, datamodule, replay_buffer, **config)
                        fq_all = f(x_q)
                        fq = fq_all.mean()

                        loss_p_x = fq - fp
                        L += config["p_x_weight"] * loss_p_x

                        if config["l2_weight"] > 0:
                            loss_l2 = (fq**2 + fp**2).mean() * config["l2_weight"]
                            L += loss_l2
                else:
                    fp_all = f(x_p_d)
                    fp = fp_all.mean()

                    x_q = sample_q(f, accelerator, datamodule, replay_buffer, **config)
                    fq_all = f(x_q)
                    fq = fq_all.mean()

                    loss_p_x = fq - fp
                    L += config["p_x_weight"] * loss_p_x

                    if config["l2_weight"] > 0:
                        loss_l2 = (fq**2 + fp**2).mean() * config["l2_weight"]
                        L += loss_l2

            """Maximize log P(y|x)"""
            if config["p_y_x_weight"] > 0:
                logits = accelerator.unwrap_model(f).classify(x_lab) if accelerator else f.classify(x_lab)
                loss_p_y_x = nn.functional.cross_entropy(logits, y_lab)
                acc = (logits.max(1)[1] == y_lab).float().mean()
                L += config["p_y_x_weight"] * loss_p_y_x

            epoch_loss += L.item()
            epoch_acc += acc.item()
            epoch_loss_p_x += loss_p_x.item() if config["p_x_weight"] > 0 else 0.0
            epoch_loss_l2 += loss_l2.item() if config["l2_weight"] > 0 else 0.0
            epoch_loss_p_y_x += loss_p_y_x.item()

            """Take gradient step"""
            optim.zero_grad()
            if accelerator:
                accelerator.backward(L)
            else:
                L.backward()
            optim.step()
            cur_iter += 1

        """---VALIDATION---"""

        f.eval()
        all_corrects, all_losses, all_confs, all_gts = [], [], [], []
        val_loss, val_acc = np.inf, 0.0
        for inputs, labels in dload_valid:
            inputs, labels = inputs.to(device), labels.to(device).squeeze().long()
            # labels = labels.squeeze().long()

            with t.no_grad():
                logits = accelerator.unwrap_model(f).classify(inputs) if accelerator else f.classify(inputs)

            if accelerator:
                losses, corrects, confs, targets = accelerator.gather_for_metrics(
                    (
                        t.nn.functional.cross_entropy(logits, labels, reduction="none"),
                        (logits.max(1)[1] == labels).float(),
                        t.nn.functional.softmax(logits, dim=1),
                        labels,
                    )
                )
            else:
                losses, corrects, confs, targets = (
                    t.nn.functional.cross_entropy(logits, labels, reduction="none"),
                    (logits.max(1)[1] == labels).float(),
                    t.nn.functional.softmax(logits, dim=1),
                    labels,
                )

            all_gts.extend(targets)
            all_confs.extend(confs)

            all_losses.extend(loss.item() for loss in losses)
            all_corrects.extend(correct.item() for correct in corrects)

        all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape((-1, datamodule.n_classes))
        all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

        val_ece = ECE(10).measure(all_confs, all_gts)
        val_loss = np.mean(all_losses)
        val_acc = np.mean(all_corrects)

        """Check if current valid loss is the best"""
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_val_ece = val_loss, val_acc, val_ece
            print_fn(f"BEST val_loss: {best_val_loss:.4f}", f"val_acc: {best_val_acc:.4f}", f"val_ece: {best_val_ece:.4f}", sep="\t")

            if accelerator:
                if accelerator.is_main_process:
                    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)

                    best_ckpt_path = f"{ckpt_dir}/epoch={epoch + (config['n_epochs'] * iter_num)}-val_loss={val_loss:.4f}.ckpt"
                    os.makedirs(ckpt_dir, exist_ok=True)

                    ckpt_dict = {
                        "model_state_dict": accelerator.unwrap_model(f).state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "replay_buffer": replay_buffer,
                    }
                    accelerator.save(ckpt_dict, best_ckpt_path)
            else:
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)

                best_ckpt_path = f"{ckpt_dir}/epoch={epoch + (config['n_epochs'] * iter_num)}-val_loss={val_loss:.4f}.ckpt"
                os.makedirs(ckpt_dir, exist_ok=True)

                ckpt_dict = {
                    "model_state_dict": accelerator.unwrap_model(f).state_dict() if accelerator else f.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "replay_buffer": replay_buffer,
                }
                t.save(ckpt_dict, best_ckpt_path)

        """---LOGGING AND CHECKPOINTING---"""

        if (epoch + (config["n_epochs"] * iter_num)) % config["sample_every_n_epochs"] == 0 and config["p_x_weight"] > 0:
            if accelerator:
                with accelerator.no_sync(f):
                    x_q = sample_q(f, datamodule, replay_buffer, accelerator=accelerator, **config)

                image = tv.utils.make_grid(x_q, normalize=True, nrow=8, value_range=(-1, 1))

                if accelerator.is_main_process:
                    if not os.path.exists(samples_dir):
                        os.makedirs(samples_dir, exist_ok=True)

                    tv.utils.save_image(
                        image,
                        f"{samples_dir}/x_q-epoch={epoch + (config['n_epochs'] * iter_num)}.png",
                    )
            else:
                x_q = sample_q(f, datamodule, replay_buffer, accelerator=accelerator, **config)

                image = tv.utils.make_grid(x_q, normalize=True, nrow=8, value_range=(-1, 1))

                if not os.path.exists(samples_dir):
                    os.makedirs(samples_dir, exist_ok=True)

                tv.utils.save_image(image, f"{samples_dir}/x_q-epoch={epoch + (config['n_epochs'] * iter_num)}.png")

        epoch_loss /= len(dload_train)
        epoch_acc /= len(dload_train)
        epoch_loss_p_x /= len(dload_train)
        epoch_loss_p_y_x /= len(dload_train)
        epoch_loss_l2 /= len(dload_train)

        print_fn(
            f"loss: {epoch_loss:.4f}",
            f"acc: {epoch_acc:.4f}",
            f"loss_p_x: {epoch_loss_p_x:.4f}",
            f"loss_l2: {epoch_loss_l2:.4f}",
            f"loss_p_y_x: {epoch_loss_p_y_x:.4f}",
            f"val_loss: {val_loss:.4f}",
            f"val_acc: {val_acc:.4f}",
            f"val_ece: {val_ece:.4f}",
            sep="\t",
        )

        if config["enable_tracking"]:
            log_values = {
                "epoch": epoch + (config["n_epochs"] * iter_num) + 1,
                "loss": epoch_loss,
                "loss_p_x": epoch_loss_p_x,
                "loss_l2": epoch_loss_l2,
                "loss_p_y_x": epoch_loss_p_y_x,
                "acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ece": val_ece,
            }
            log_fn(log_values)

    """Log the final epoch"""
    if config["enable_tracking"]:
        log_fn(
            {
                "num_labeled": len(datamodule.train_labeled_indices),
                "loss": epoch_loss,
                "loss_p_x": epoch_loss_p_x,
                "loss_l2": epoch_loss_l2,
                "loss_p_y_x": epoch_loss_p_y_x,
                "acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ece": val_ece,
            }
        )

    """Log the best epoch"""
    if config["enable_tracking"]:
        log_fn(
            {
                "num_labeled": len(datamodule.train_labeled_indices),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_val_ece": best_val_ece,
            }
        )

    """Save the last checkpoint"""
    ckpt_dict = {
        "model_state_dict": accelerator.unwrap_model(f).state_dict() if accelerator else f.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "replay_buffer": replay_buffer,
    }

    if accelerator:
        if accelerator.is_main_process:
            accelerator.save(ckpt_dict, f"{ckpt_dir}/last.ckpt")
    else:
        t.save(ckpt_dict, f"{ckpt_dir}/last.ckpt")

    return f


def init_logger(experiment_name: str, experiment_type: str, log_dir: str, num_labeled: int = None):
    dir_name = f"{experiment_type}_{num_labeled}"
    run_name = experiment_type

    logger_kwargs = {"group": experiment_name, "name": run_name}
    ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints", dir_name)
    samples_dir = os.path.join(log_dir, experiment_name, "samples", dir_name)
    test_dir = os.path.join(log_dir, experiment_name, "test", dir_name)

    return logger_kwargs, (ckpt_dir, samples_dir, test_dir)


limit_dict = {
    "cifar10": 40000,
    "cifar100": 40000,
    "svhn": 40000,
    "bloodmnist": 4000,
    "dermamnist": 4000,
    "pneumoniamnist": 4000,
    "organsmnist": 4000,
    "organcmnist": 4000,
}

equal_dict = {
    "pneumoniamnist": 2400,
    "bloodmnist": 6400,
}


def main(config):
    accelerator = Accelerator(log_with="wandb" if config["enable_tracking"] else None) if config["multi_gpu"] else None
    print_fn = accelerator.print if accelerator else print
    device = accelerator.device if accelerator else t.device("cuda")
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
        sampling_method="random" if config["labels_per_class"] <= 0 else None,
        init_size=config["query_size"],
        labels_per_class=config["labels_per_class"],
        accelerator=accelerator,
    )

    """For informative initialization"""
    if not os.path.isfile(f"weights/{datamodule.dataset}_cov.pt"):
        category_mean(dload_train=dload_train, datamodule=datamodule)

    n_iters = len(datamodule.full_train) // config["query_size"]
    limit = limit_dict[config["dataset"]] if config["labels_per_class"] <= 0 else equal_dict[config["dataset"]]
    experiment_name = get_experiment_name(**config)
    init_size = config["query_size"]
    labels_per_class = config["labels_per_class"]

    for i in range(n_iters):
        raw_f, replay_buffer = get_model_and_buffer(datamodule=datamodule, accelerator=accelerator, device=device, **config)
        replay_buffer = init_from_centers(device=device, datamodule=datamodule, **config)
        optim = get_optimizer(raw_f, accelerator=accelerator, **config)
        logger_kwargs, dirs = init_logger(experiment_name, config["experiment_type"], config["log_dir"], len(train_labeled_inds))

        if config["enable_tracking"]:
            if accelerator:
                accelerator.init_trackers(project_name="JEM", config=config, init_kwargs={"wandb": logger_kwargs})
            else:
                wandb.init(project="JEM", config=config, **logger_kwargs)

        """
            ---TRAINING---
            - Train the model.
        """
        trained_f = train_model(
            f=raw_f,
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

        if len(train_labeled_inds) >= limit:
            print_fn(f"Training complete with {len(train_labeled_inds)} labeled samples.")
            break

        # Least confident sampling
        if config["labels_per_class"] == 0:
            print_fn(f"Querying {config['query_size']} samples using least confident sampling.")
            inds_to_fix = datamodule.query_samples(
                trained_f,
                dload_train_unlabeled,
                train_unlabeled_inds,
                config["query_size"],
                accelerator=accelerator,
            )
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(
                train_labeled_inds,
                train_unlabeled_inds,
                inds_to_fix,
                start_iter=False,
                accelerator=accelerator,
            )

        # Equal labels sampling
        elif config["labels_per_class"] > 0:
            print_fn(f"Querying {config['labels_per_class']} samples per class.")
            labels_per_class += config["labels_per_class"]
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(
                train_labeled_inds,
                train_unlabeled_inds,
                sampling_method="equal",
                labels_per_class=labels_per_class,
                accelerator=accelerator,
            )

        # Random sampling
        else:
            print_fn(f"Querying {config['query_size']} samples randomly.")
            init_size += config["query_size"]
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(
                train_labeled_inds,
                train_unlabeled_inds,
                sampling_method="random",
                init_size=init_size,
                accelerator=accelerator,
            )

    if config["enable_tracking"]:
        if accelerator:
            accelerator.end_training()
        else:
            wandb.finish()


if __name__ == "__main__":
    t.backends.cudnn.enabled = True
    t.backends.cudnn.deterministic = True

    config = vars(parse_args())

    """Scale batch size by number of GPUs for reproducibility"""
    config.update({"p_x_weight": 1.0 if config["calibrated"] else 0.0})
    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})
    config.update({"experiment_name": config["dataset"]})

    set_seed(config["seed"])

    main(config)
