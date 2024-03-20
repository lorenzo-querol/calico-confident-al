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
import shutil
from temperature_scaling import ModelWithTemperature
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from accelerate.utils import set_seed
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from tensorboardX import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import get_model, get_optim
from utils import Hamiltonian, parse_args, write_to_yaml, create_log_dir

conditionals = []


def init_random(datamodule: DataModule):
    global conditionals

    batch_size = datamodule.batch_size
    n_classes = datamodule.n_classes
    img_shape = datamodule.img_shape

    new = t.zeros(batch_size, img_shape[0], img_shape[1], img_shape[2])

    for i in range(batch_size):
        index = np.random.randint(n_classes)
        dist = conditionals[index]
        new[i] = dist.sample().view(img_shape)

    return t.clamp(new, -1, 1).cpu()


def init_from_centers(datamodule, args, device):
    global conditionals

    centers = t.load(f"weights/{args.dataset}_mean.pt")
    covs = t.load(f"weights/{args.dataset}_cov.pt")

    buffer = []
    samples_per_class = datamodule.batch_size // datamodule.n_classes

    for i in range(datamodule.n_classes):
        mean, cov = centers[i].to(device), covs[i].to(device)
        cov += 1e-4 * t.eye(int(np.prod(datamodule.img_shape))).to(device)
        dist = MultivariateNormal(mean, covariance_matrix=cov)

        conditionals.append(dist)

        tensor = dist.sample((samples_per_class,)).view((samples_per_class,) + datamodule.img_shape).cpu()

        buffer.append(tensor)

    images = t.clamp(t.cat(buffer), -1, 1)

    return images


def category_mean(datamodule: DataModule):
    img_shape = datamodule.img_shape
    n_classes = datamodule.n_classes
    train_dataloader = datamodule.full_train

    centers = t.zeros([n_classes, int(np.prod(img_shape))])
    covs = t.zeros([n_classes, int(np.prod(img_shape)), int(np.prod(img_shape))])

    for i in range(n_classes):
        im_class = []
        for im, targ in train_dataloader:
            mask = (targ == i).squeeze(1)
            imc = im[mask]
            imc = imc.view(len(imc), -1)
            im_class.append(imc)

        im_class = t.cat(im_class)
        mean = im_class.mean(dim=0)
        sub = im_class - mean.unsqueeze(dim=0)
        cov = sub.t() @ sub / len(im_class)
        centers[i] = mean
        covs[i] = cov

    if not os.path.exists("weights"):
        os.makedirs("./weights")

    t.save(centers, f"weights/{datamodule.dataset}_mean.pt")
    t.save(covs, f"weights/{datamodule.dataset}_cov.pt")


def sample_p_0(replay_buffer, datamodule, args, device):
    if len(replay_buffer) == 0:
        return init_random(datamodule), []

    indices = t.randint(0, len(replay_buffer), (args.batch_size,))

    buffer_samples = replay_buffer[indices]
    random_samples = init_random(datamodule)
    choose_random = (t.rand(args.batch_size) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples

    return samples.to(device), indices


def sample_q(f, datamodule, replay_buffer, args, device):
    init_sample, buffer_indices = sample_p_0(replay_buffer, datamodule, args, device)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)

    in_steps = args.in_steps if args.pyld_lr > 0 else 0

    Hamiltonian_func = Hamiltonian(f.f.layer_one) if in_steps > 0 else None

    for _ in range(args.n_steps):
        energies = f(x_k)
        e_x = energies.sum()
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]

        p = f.f.layer_one_out.grad.detach() if in_steps > 0 else None

        tmp_inp = x_k.data
        tmp_inp.requires_grad_()

        for _ in range(in_steps):
            H = Hamiltonian_func(tmp_inp, p)

            eta_grad = t.autograd.grad(H, [tmp_inp], only_inputs=True, retain_graph=True)[0]
            eta_step = t.clamp(eta_grad, -args.eps, args.eps) * args.pyld_lr

            tmp_inp.data = tmp_inp.data + eta_step
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        x_k.data = tmp_inp.data

    if in_steps > 0:
        loss = -1.0 * Hamiltonian_func(x_k.data, p)
        loss.backward()

    f.train()
    final_samples = x_k.detach()

    replay_buffer[buffer_indices] = final_samples.cpu()

    return final_samples


def compute_ece(confs, gts, n_classes, save_rel_diagram=False, save_path=None):
    """
    Computes the Expected Calibration Error (ECE) of a model.

    #### params:
    - confs: numpy array of shape (n_samples, n_classes)
        - The confidence scores for each sample.
    - gts: numpy array of shape (n_samples,)
        - The ground truth labels for each sample.
    - n_classes: int
        - The number of classes in the dataset.

    #### returns:
    - ece: float
        - The Expected Calibration Error (ECE) of the model.
    """

    BINS = 10

    confs, gts = np.concatenate(confs), np.concatenate(gts)
    all_confs = confs.reshape((-1, n_classes))

    ece = ECE(BINS).measure(all_confs, gts)

    if save_rel_diagram:
        assert save_path is not None, "save_path must be provided if save_rel_diagram is True."
        ReliabilityDiagram(BINS).plot(all_confs, gts, filename=save_path, tikz=True)
        ReliabilityDiagram(BINS).plot(all_confs, gts, filename=save_path.replace(".tikz", ".png"))

    return ece


def fit(f: nn.Module, datamodule: DataModule, args: argparse.Namespace, writer: SummaryWriter, log_dir: str, al_iter: int):

    # Informative initialization
    if not os.path.isfile(f"weights/{datamodule.dataset}_cov.pt"):
        category_mean(datamodule=datamodule)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    f.to(device)

    optim = get_optim(f, args)

    samples_dir = f"{log_dir}/samples"
    ckpt_dir = f"{log_dir}/checkpoints"
    test_dir = f"{log_dir}/test"

    for dir in [ckpt_dir, test_dir, samples_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    progress_bar = tqdm(range(args.n_epochs), desc="Training Progress", total=args.n_epochs, position=0, leave=True)

    train_dataloader = datamodule.train_dataloader()
    labeled_dataloader = datamodule.labeled_dataloader()
    val_dataloader = datamodule.val_dataloader()

    replay_buffer = init_from_centers(datamodule, args, device)

    curr_iter = 0
    for epoch in progress_bar:
        # Decay LR
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group["lr"] * args.decay_rate
                param_group["lr"] = new_lr

        batch_prog_bar = tqdm(
            zip(train_dataloader, labeled_dataloader),
            desc="Batch Progress",
            total=len(train_dataloader),
            position=1,
            leave=False,
        )

        best_val_loss = float("inf")

        # Training
        f.train()
        train_loss, train_acc = [], []
        for train_batch, labeled_batch in batch_prog_bar:

            # Warmup LR
            if curr_iter <= args.warmup_iters:
                for param_group in optim.param_groups:
                    lr = args.lr * curr_iter / float(args.warmup_iters)
                    param_group["lr"] = lr

            L, acc = 0.0, 0.0

            # Optimize log p(x)
            if args.px > 0:
                x_p_d, _ = train_batch
                x_p_d = x_p_d.to(device)

                fp_all = f(x_p_d)
                fp = fp_all.mean()

                x_q = sample_q(f, datamodule, replay_buffer, args, device)

                fq_all = f(x_q)
                fq = fq_all.mean()

                cdiv_loss = -(fp - fq)

                L += args.px * cdiv_loss

            # Optimize log p(y|x)
            if args.pyx > 0:
                x_lab, y_lab = labeled_batch
                x_lab, y_lab = x_lab.to(device), y_lab.to(device).squeeze().long()

                logits = f.classify(x_lab)
                ce_loss = args.pyx * F.cross_entropy(logits, y_lab)
                acc = (logits.max(1)[1] == y_lab).float().mean()

                L += args.pyx * ce_loss

            train_loss.append(L)
            train_acc.append(acc)

            optim.zero_grad()
            L.backward()
            optim.step()
            curr_iter += 1

        # Average loss and accuracy over the epoch
        train_loss = t.mean(t.stack(train_loss))
        train_acc = t.mean(t.stack(train_acc))

        # Validation
        f.eval()
        val_loss, val_acc, all_confs, all_gts = [], [], [], []
        for x_lab, y_lab in val_dataloader:
            x_lab, y_lab = x_lab.to(device), y_lab.to(device).squeeze().long()

            with t.no_grad():
                logits = f.classify(x_lab)
                ce_loss = F.cross_entropy(logits, y_lab, reduction="none").detach().cpu().numpy()
                confs = F.softmax(logits, dim=1).float().cpu().numpy()

            acc = (logits.max(1)[1] == y_lab).float().cpu().numpy()
            gts = y_lab.detach().cpu().numpy()

            val_loss.extend(ce_loss)
            val_acc.extend(acc)
            all_confs.extend(confs)
            all_gts.append(gts)

        # Average loss and accuracy over the validation set
        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)
        val_ece = compute_ece(all_confs, all_gts, datamodule.n_classes)

        # Save best model if validation loss is lower
        if val_loss < best_val_loss:
            t.save(f.state_dict(), f"{ckpt_dir}/num-labels-{len(datamodule.labeled_indices)}_best.ckpt")
            best_val_loss = val_loss

        if epoch % args.sample_every_n_epochs == 0 and args.px > 0:
            samples = sample_q(f, datamodule, replay_buffer, args, device)
            filename = f"{samples_dir}/epoch_{al_iter * args.n_epochs + epoch}.jpg"
            tv.utils.save_image(samples, filename, normalize=True, value_range=(-1, 1))

        metrics = {
            "train/loss": train_loss.item(),
            "train/acc": train_acc.item(),
            "val/loss": val_loss.item(),
            "val/acc": val_acc.item(),
            "val/ece": val_ece,
            "lr": optim.param_groups[0]["lr"],
        }

        # Log to tensorboard
        for key, value in metrics.items():
            writer.add_scalar(key, value, al_iter * args.n_epochs + epoch)

        writer.flush()

        # Log to progress bar
        for key, value in metrics.items():
            metrics[key] = "{:.4f}".format(value)

        progress_bar.set_postfix(metrics)

    # Log last epoch
    metrics = {
        "al-iter-train/loss": train_loss.item(),
        "al-iter-train/acc": train_acc.item(),
        "al-iter-val/loss": val_loss.item(),
        "al-iter-val/acc": val_acc.item(),
        "al-iter-val/ece": val_ece,
    }

    for key, value in metrics.items():
        writer.add_scalar(key, value, len(datamodule.labeled_indices))

    # Save last model checkpoint
    t.save(f.state_dict(), f"{ckpt_dir}/num-labels-{len(datamodule.labeled_indices)}_last.ckpt")


def test(f, datamodule: DataModule, path: str, num_labeled: int):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    f.to(device)

    test_dataloader = datamodule.test_dataloader()

    progress_bar = tqdm(test_dataloader, desc="Test Progress", total=len(test_dataloader), position=0, leave=True)

    f.eval()
    test_loss, test_acc, all_confs, all_gts = [], [], [], []
    for x_lab, y_lab in progress_bar:
        x_lab, y_lab = x_lab.to(device), y_lab.to(device).squeeze().long()

        with t.no_grad():
            logits = f.classify(x_lab)
            ce_loss = nn.functional.cross_entropy(logits, y_lab, reduction="none").detach().cpu().numpy()
            confs = nn.functional.softmax(logits, dim=1).float().cpu().numpy()

        acc = (logits.max(1)[1] == y_lab).float().cpu().numpy()
        gts = y_lab.detach().cpu().numpy()

        test_loss.extend(ce_loss)
        test_acc.extend(acc)
        all_confs.extend(confs)
        all_gts.append(gts)

    # Average loss and accuracy over the test set
    test_loss = np.mean(test_loss)
    test_acc = np.mean(test_acc)
    test_ece = compute_ece(
        all_confs,
        all_gts,
        datamodule.n_classes,
        save_rel_diagram=True,
        save_path=f"{path}/num-labels-{num_labeled}-reliability.tikz",
    )

    # Save as a CSV file
    test_df = pd.DataFrame(
        {
            "num_labeled": [num_labeled],
            "test_loss": [test_loss.item()],
            "test_acc": [test_acc.item()],
            "test_ece": [test_ece],
        }
    )

    if os.path.exists(f"{path}/test_metrics.csv"):
        test_df.to_csv(f"{path}/test_metrics.csv", mode="a", header=False, index=False)
    else:
        test_df.to_csv(f"{path}/test_metrics.csv", mode="w", header=True, index=False)


MEDMNIST_DATASETS = ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]
LIMIT_DICTS = {"bloodmnist": 4000, "organcmnist": 3850, "organsmnist": 3850, "pneumoniamnist": 2000}


def train_model(args):
    datamodule = DataModule(dataset=args.dataset, root_dir=args.root_dir, batch_size=args.batch_size, sigma=args.sigma)

    LIMIT = 4000
    iterations = LIMIT // args.query_size

    log_dir = f"./{args.log_dir}/{datamodule.dataset}/{args.exp_name}"
    log_dir = create_log_dir(log_dir)

    writer = SummaryWriter(log_dir)

    write_to_yaml(vars(args), f"{log_dir}/config.yaml")

    # Baseline experiments
    if args.exp_name == "baseline-softmax":
        print("Dataset:", args.dataset)
        print(f"\n|---Training baseline model with {args.query_size} labeled samples---|")
        datamodule.setup(sample_method=args.sample_method, init_size=args.query_size, log_dir=log_dir)

        model = get_model(datamodule, args)
        fit(model, datamodule, args, writer, log_dir, 0)

    # Equal class distribution experiments
    elif args.exp_name == "equal-jempp" and args.sample_method == "equal":

        LIMIT = LIMIT_DICTS[args.dataset]
        iterations = LIMIT // (args.labels_per_class * datamodule.n_classes)
        assert LIMIT % (args.labels_per_class * datamodule.n_classes) == 0, "The number of labeled samples must be divisible by the query size."
        labels_per_class = args.labels_per_class

        datamodule.setup(sample_method=args.sample_method, labels_per_class=labels_per_class, log_dir=log_dir)

        for i in range(iterations):
            print("Dataset:", args.dataset)
            print(f"\n|---Iteration {i+1}/{iterations}: training with {labels_per_class} labeled samples per class---|")

            model = get_model(datamodule, args)
            fit(model, datamodule, args, writer, log_dir, i)

            if len(datamodule.labeled_indices) != LIMIT:
                labels_per_class += args.labels_per_class
                datamodule.setup(sample_method=args.sample_method, labels_per_class=labels_per_class, log_dir=log_dir)

    # Active learning experiments
    else:
        datamodule.setup(sample_method=args.sample_method, init_size=args.query_size, log_dir=log_dir)

        for i in range(iterations):
            print("Dataset:", args.dataset)
            print(f"\n|---Active Learning Iteration {i+1}/{iterations}---|")

            model = get_model(datamodule, args)
            fit(model, datamodule, args, writer, log_dir, i)

            if len(datamodule.labeled_indices) != LIMIT:
                indices_to_fix = datamodule.query(model, args.query_size, log_dir)
                datamodule.setup(indices_to_fix=indices_to_fix, start_iter=False, log_dir=log_dir)

    writer.close()


def test_model(args):
    datasets = os.walk(args.log_dir).__next__()[1]

    for dataset in datasets:
        args.dataset = dataset
        exp_types = os.walk(f"{args.log_dir}/{args.dataset}").__next__()[1]
        datamodule = DataModule(dataset=args.dataset, root_dir=args.root_dir, batch_size=args.batch_size, sigma=args.sigma)

        for exp_type in exp_types:
            if args.temp_scale and exp_type in ["baseline-softmax", "active-jempp", "equal-jempp"]:
                continue

            type = exp_type

            if args.temp_scale:
                type = type + "-temp_scaled"

            TEST_PATH = f"{args.test_dir}/{datamodule.dataset}/{type}"

            if os.path.exists(TEST_PATH):
                shutil.rmtree(TEST_PATH, ignore_errors=True)
                os.makedirs(TEST_PATH, exist_ok=True)
            else:
                os.makedirs(TEST_PATH, exist_ok=True)

            datamodule.test_setup(TEST_PATH)

            CKPT_DIR = f"{args.log_dir}/{args.dataset}/{exp_type}/checkpoints"
            ckpts = [name for name in os.listdir(CKPT_DIR) if args.ckpt_type in name]
            ckpts = [{"num_labeled": int(ckpt.split("_")[0].split("-")[-1]), "path": f"{CKPT_DIR}/{ckpt}"} for ckpt in ckpts]
            ckpts = sorted(ckpts, key=lambda x: x["num_labeled"])

            for ckpt in ckpts:
                print(f"\n|---Testing Model with {ckpt['num_labeled']} Labeled Samples---|")
                model = get_model(datamodule, args, ckpt["path"])

                if args.temp_scale:
                    model = ModelWithTemperature(model)
                    model.set_temperature(datamodule.val_dataloader())

                test(model.model if args.temp_scale else model, datamodule, TEST_PATH, ckpt["num_labeled"])


if __name__ == "__main__":
    t.backends.cudnn.enabled = True
    t.backends.cudnn.deterministic = True

    args = parse_args()
    set_seed(args.seed)

    if not args.test:
        assert args.exp_name is not None, "Experiment name must be provided."

    test_model(args) if args.test else train_model(args)
