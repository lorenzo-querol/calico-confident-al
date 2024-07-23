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

import numpy as np
import torch as t
import torch.nn as nn
from netcal.metrics import ECE
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from DataModule import DataModule
from models.JEM import get_model_and_buffer, get_optimizer
from utils import Hamiltonian, create_log_dir, initialize, parse_args, plot

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

    if load_path:
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


def sample_q(f, datamodule, replay_buffer, batch_size, n_steps, in_steps, sgld_std, sgld_lr, pyld_lr, eps, y=None, save=True, **config):
    bs = batch_size

    init_sample, buffer_inds = sample_p_0(replay_buffer=replay_buffer, datamodule=datamodule, bs=bs, y=y, **config)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)

    if in_steps > 0:
        Hamiltonian_func = Hamiltonian(f.f.layer_one)

    if pyld_lr <= 0:
        in_steps = 0

    for it in range(n_steps):
        energies = f(x_k)
        e_x = energies.sum()
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]

        if in_steps > 0:
            p = 1.0 * f.f.layer_one_out.grad
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


def fit(
    f: nn.Module,
    optim: t.optim.Optimizer,
    datamodule: DataModule,
    dload_train: DataLoader,
    dload_train_labeled: DataLoader,
    dload_valid: DataLoader,
    replay_buffer: t.Tensor,
    log_dir: str,
    device: str,
    al_iter: int = 0,
    **config,
):
    samples_dir = f"{log_dir}/samples"
    ckpt_dir = f"{log_dir}/checkpoints"
    test_dir = f"{log_dir}/test"

    for dir in [ckpt_dir, test_dir, samples_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    cur_iter = 0
    new_lr = config["lr"]
    best_val_loss, best_val_acc, best_val_ece = np.inf, 0.0, np.inf
    best_ckpt_path = None

    for epoch in range(config["n_epochs"]):
        if epoch in config["decay_epochs"]:
            for param_group in optim.param_groups:
                new_lr = param_group["lr"] * config["decay_rate"]
                param_group["lr"] = new_lr

            print(f"Decaying LR to {new_lr:.8f}.")

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_l_px = 0.0
        epoch_l_pyx = 0.0
        epoch_l_l2 = 0.0
        l_px = 0.0
        l_l2 = 0.0

        # Training
        f.train()
        for i, (x_p_d, _) in enumerate(tqdm(dload_train, desc=(f"Epoch {epoch+1}/{config['n_epochs']}"))):
            """Warmup Learning Rate"""
            if cur_iter <= config["warmup_iters"]:
                lr = config["lr"] * cur_iter / float(config["warmup_iters"])
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = (x_lab.to(device), y_lab.to(device).squeeze().long())

            L = 0.0

            """Maximize log P(x)"""
            if config["px"] > 0:
                fp_all = f(x_p_d)
                fp = fp_all.mean()

                x_q = sample_q(f, datamodule, replay_buffer, **config)
                fq_all = f(x_q)
                fq = fq_all.mean()

                l_px = fq - fp
                L += config["px"] * l_px

                if config["l2"] > 0:
                    l_l2 = (fq**2 + fp**2).mean() * config["l2"]
                    L += l_l2

            """Maximize log P(y|x)"""
            if config["pyx"] > 0:
                logits = f.classify(x_lab)
                l_pyx = nn.functional.cross_entropy(logits, y_lab)
                acc = (logits.max(1)[1] == y_lab).float().mean()
                L += config["pyx"] * l_pyx

            epoch_loss += L.item()
            epoch_acc += acc.item()
            epoch_l_px += l_px.item() if config["px"] > 0 else 0.0
            epoch_l_l2 += l_l2.item() if config["l2"] > 0 else 0.0
            epoch_l_pyx += l_pyx.item()

            """Take gradient step"""
            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

        # Validation
        f.eval()
        all_corrects, all_losses, all_confs, all_gts = [], [], [], []
        val_loss, val_acc = np.inf, 0.0
        for inputs, labels in dload_valid:
            inputs, labels = inputs.to(device), labels.to(device).squeeze().long()

            with t.no_grad():
                logits = f.classify(inputs)

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

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_val_ece = val_loss, val_acc, val_ece

            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)

            ckpt_dict = {
                "model_state_dict": f.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "replay_buffer": replay_buffer,
            }
            t.save(ckpt_dict, f"{ckpt_dir}/al_iter={al_iter}-epoch={epoch}-val_loss_{val_loss:.4f}.ckpt")

        # Logging

        if (epoch + (config["n_epochs"] * al_iter)) % config["sample_every_n_epochs"] == 0 and config["px"] > 0:
            x_q = sample_q(f, datamodule, replay_buffer, **config)
            plot(f"{samples_dir}/x_q-al_iter={al_iter}-epoch={epoch}.png", x_q)

        epoch_loss /= len(dload_train)
        epoch_acc /= len(dload_train)
        epoch_l_px /= len(dload_train)
        epoch_l_pyx /= len(dload_train)
        epoch_l_l2 /= len(dload_train)

        print(
            f"loss: {epoch_loss:.4f}",
            f"acc: {epoch_acc:.4f}",
            f"l_px: {epoch_l_px:.4f}",
            f"l_pyx: {epoch_l_pyx:.4f}",
            f"l_l2: {epoch_l_l2:.4f}",
            f"val_loss: {val_loss:.4f}",
            f"val_acc: {val_acc:.4f}",
            f"val_ece: {val_ece:.4f}",
            sep="\t",
        )

        if config["enable_tracking"]:
            log_values = {
                "epoch": epoch + (config["n_epochs"] * al_iter) + 1,
                "loss": epoch_loss,
                "l_px": epoch_l_px,
                "l_pyx": epoch_l_pyx,
                "l_l2": epoch_l_l2,
                "acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ece": val_ece,
            }
            wandb.log(log_values)

    """Log the final epoch"""
    if config["enable_tracking"]:
        wandb.log(
            {
                "num_labeled": len(datamodule.train_labeled_indices),
                "loss": epoch_loss,
                "l_px": epoch_l_px,
                "l_pyx": epoch_l_pyx,
                "l_l2": epoch_l_l2,
                "acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ece": val_ece,
            }
        )

    """Log the best epoch"""
    if config["enable_tracking"]:
        wandb.log(
            {
                "num_labeled": len(datamodule.train_labeled_indices),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_val_ece": best_val_ece,
            }
        )

    """Save the last checkpoint"""
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "replay_buffer": replay_buffer,
    }

    t.save(ckpt_dict, f"{ckpt_dir}/al_iter={al_iter}-epoch={epoch}-last.ckpt")

    return f


limit_dict = {
    "bloodmnist": 4000,
    "dermamnist": 4000,
    "pneumoniamnist": 4000,
    "organsmnist": 4000,
    "organcmnist": 4000,
}

equal_dict = {
    "pneumoniamnist": 2400,  # labels per class 50 / 12 iterations
    "bloodmnist": 4000,  # labels per class 50 / 10 iterations (4000)
    "organcmnist": 3850,  # labels per class 35 / 10 iterations (3850)
    "organsmnist": 3850,  # labels per class 35 / 10 iterations (3850)
}


def main(config):
    initialize(config["seed"])

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    datamodule = DataModule(**config)
    datamodule.prepare_data()

    (
        dload_train,
        dload_train_labeled,
        dload_train_unlabeled,
        dload_valid,
        train_labeled_inds,
        train_unlabeled_inds,
    ) = datamodule.get_data(
        sample_method="random" if config["labels_per_class"] <= 0 else "equal",
        init_size=config["query_size"],
        labels_per_class=config["labels_per_class"],
    )

    # Informative initialization
    if not os.path.isfile(f"weights/{datamodule.dataset}_cov.pt"):
        category_mean(dload_train=dload_train, datamodule=datamodule)

    n_iters = len(datamodule.full_train) // config["query_size"]
    limit = limit_dict[config["dataset"]] if config["labels_per_class"] <= 0 else equal_dict[config["dataset"]]
    init_size = config["query_size"]
    labels_per_class = config["labels_per_class"]

    log_dir = f"./{config['log_dir']}/{config['dataset']}/{config['exp_name']}"
    log_dir = create_log_dir(log_dir)
    config.update({"log_dir": log_dir})

    for i in range(n_iters):
        raw_f, replay_buffer = get_model_and_buffer(datamodule, device=device, **config)
        replay_buffer = init_from_centers(device=device, datamodule=datamodule, **config)
        optim = get_optimizer(raw_f, device=device, **config)

        if config["enable_tracking"]:
            wandb.init(project="CALICO", config=config, group=config["dataset"], name=config["exp_name"])

        trained_f = fit(
            f=raw_f,
            optim=optim,
            datamodule=datamodule,
            dload_train=dload_train,
            dload_train_labeled=dload_train_labeled,
            dload_valid=dload_valid,
            train_labeled_inds=train_labeled_inds,
            replay_buffer=replay_buffer,
            device=device,
            al_iter=i,
            **config,
        )

        if len(train_labeled_inds) >= limit:
            print(f"Training complete with {len(train_labeled_inds)} labeled samples.")
            break

        # Least confident sampling
        if config["labels_per_class"] == 0:
            print(f"Querying {config['query_size']} samples using least confident sampling.")
            inds_to_fix = datamodule.query_samples(
                trained_f,
                dload_train_unlabeled,
                train_unlabeled_inds,
                config["query_size"],
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
            )

        # Equal labels sampling
        elif config["labels_per_class"] > 0:
            print(f"Querying {config['labels_per_class']} samples per class.")
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
                sample_method="equal",
                labels_per_class=labels_per_class,
            )

        # Random sampling
        else:
            print(f"Querying {config['query_size']} samples randomly.")
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
                sample_method="random",
                init_size=init_size,
            )

    if config["enable_tracking"]:
        wandb.finish()


if __name__ == "__main__":
    config = vars(parse_args())
    main(config)
