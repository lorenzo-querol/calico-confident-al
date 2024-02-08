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
import torchvision as tv
from accelerate import Accelerator
from accelerate.utils import set_seed
from netcal.metrics import ECE
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import F
from utils import get_args

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


def init_from_centers(datamodule: DataModule, buffer_size: int, device: str, **config):
    global conditionals

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


def sample_q(f, datamodule, replay_buffer, batch_size, n_steps, sgld_std, sgld_lr, y=None, save=True, accelerator=None, **config):
    bs = batch_size

    init_sample, buffer_inds = sample_p_0(replay_buffer=replay_buffer, datamodule=datamodule, bs=bs, y=y, **config)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)

    for it in range(n_steps):
        energies = f(x_k, y=y)
        e_x = energies.sum()
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        tmp_inp = x_k + t.clamp(eta, -1, 1) * sgld_lr

        x_k.data = tmp_inp.data

        if sgld_std > 0.0:
            x_k.data += sgld_std * t.randn_like(x_k)
        x_k.data = t.clamp(x_k.data, -1, 1)

    f.train()
    final_samples = x_k.detach()

    if len(replay_buffer) > 0 and save:
        replay_buffer[buffer_inds] = final_samples.cpu()

    return final_samples


def category_mean(dm: DataModule):
    dataset = dm.dataset
    img_shape = dm.img_shape
    n_classes = dm.n_classes
    train_dataloader = dm.get_train_dataloader()

    centers = t.zeros([n_classes, int(np.prod(img_shape))])
    covs = t.zeros([n_classes, int(np.prod(img_shape)), int(np.prod(img_shape))])

    im_test, targ_test = [], []
    for im, targ in train_dataloader:
        im_test.append(im)
        targ_test.append(targ)

    im_test, targ_test = t.cat(im_test), t.cat(targ_test)

    for i in range(n_classes):
        if dm.dataset in ["cifar10", "cifar100", "svhn", "mnist"]:
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
        os.makedirs("./weights")

    t.save(centers, f"weights/{dataset}_mean.pt")
    t.save(covs, f"weights/{dataset}_cov.pt")


class CustomTrainer:
    def __init__(self, f: nn.Module, opt: t.optim.Optimizer, datamodule: DataModule, replay_buffer, device, args):
        self.f = f
        self.opt = opt
        self.datamodule = datamodule
        self.replay_buffer = replay_buffer
        self.device = device
        self.hparams = args

        self.train_dataloader = self.datamodule.get_train_dataloader()
        self.labeled_dataloader = self.datamodule.get_labeled_dataloader()
        self.val_dataloader = self.datamodule.get_val_dataloader()

    def fit(self):
        raise NotImplementedError

    def _calculate_px_loss(self, batch):
        x_p_d, _ = batch
        L = 0.0

        fp_all = self.f(x_p_d)
        fp = fp_all.mean()
        x_q = sample_q(self.f, self.datamodule, self.replay_buffer, accelerator=self.accelerator, **self.config)
        fq_all = self.f(x_q)
        fq = fq_all.mean()

        cdiv_loss = -(fp - fq)
        L += self.hparams.px * cdiv_loss

        return L

    def _calculate_pyx_loss(self, batch):
        x_lab, y_lab = batch
        L = 0.0

        logits = self.f.classify(x_lab)
        ce_loss = nn.functional.cross_entropy(logits, y_lab.squeeze().long())
        acc = (logits.max(1)[1] == y_lab).float().mean()
        L += self.hparams.pyx * ce_loss

        return L, acc

    def _training_step(self, train_batch, labeled_batch):
        loss, acc = 0.0, 0.0

        if self.hparams.px > 0:
            loss = self._calculate_px_loss(train_batch)

        if self.hparams.pyx > 0:
            ce_loss, acc = self._calculate_pyx_loss(labeled_batch)
            loss += ce_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss, acc

    def _valid_step(self, batch):
        inputs, labels = batch

        logits = self.f.classify(inputs)
        loss = nn.functional.cross_entropy(logits, labels.squeeze().long())
        acc = (logits.max(1)[1] == labels).float().mean()

        return loss, acc

    def _init_trackers(self):
        self.current_epoch = 0
        self.current_iter = 0
        self.best_val_loss = np.inf
        self.best_val_acc = 0.0
        self.best_val_ece = 0.0

    def _decay_lr(self):
        if self.current_epoch in self.hparams.decay_epochs:
            for param_group in self.opt.param_groups:
                new_lr = param_group["lr"] * self.hparams.decay_rate
                param_group["lr"] = new_lr

    def _warmup_lr(self):
        if self.current_iter <= self.hparams.warmup_iters:
            lr = self.hparams.lr * self.current_iter / float(self.hparams.warmup_iters)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

    def fit(self):
        self._init_trackers()

        for epoch in range(self.hparams.n_epochs):
            self.current_epoch = epoch
            self._decay_lr()

            progress_bar = tqdm(
                zip(self.train_dataloader, self.labeled_dataloader),
                desc=(f"Epoch [{self.current_epoch+1}/{self.hparams.n_epochs}]"),
                total=len(self.datamodule.full_train),
            )

            self.f.train()
            for i, (train_batch, labeled_batch) in enumerate(progress_bar):
                self.current_iter += 1
                self._warmup_lr()
                loss, acc = self._training_step(train_batch, labeled_batch)
                progress_bar.set_postfix(
                    {
                        "lr": self.opt.param_groups[0]["lr"],
                        "loss": loss.item(),
                        "acc": acc.item(),
                    }
                )

            self.f.eval()
            val_loss, val_acc = 0, 0
            for batch in self.val_dataloader:
                loss, acc = self._valid_step(batch)
                val_loss += loss
                val_acc += acc

                val_loss /= len(self.val_dataloader)
                val_acc /= len(self.val_dataloader)
                progress_bar.set_postfix(
                    {
                        "lr": self.opt.param_groups[0]["lr"],
                        "loss": loss.item(),
                        "acc": acc.item(),
                        "val_loss": val_loss.item(),
                        "val_acc": val_acc.item(),
                    }
                )


# def train_model(
#     f: nn.Module,
#     optim: t.optim.Optimizer,
#     datamodule: DataModule,
#     dload_train: DataLoader,
#     dload_train_labeled: DataLoader,
#     dload_valid: DataLoader,
#     replay_buffer: t.Tensor,
#     dirs: tuple[str, str, str],
#     accelerator: Accelerator = None,
#     **config,
# ):
#     device = accelerator.device if accelerator else t.device("cuda")
#     ckpt_dir, samples_dir, _ = dirs

#     cur_iter = 0
#     new_lr = config["lr"]
#     best_val_loss, best_val_acc, best_val_ece = np.inf, 0.0, 0.0
#     best_ckpt_path = None

#     for epoch in range(config["n_epochs"]):
#         if epoch in config["decay_epochs"]:
#             for param_group in optim.param_groups:
#                 new_lr = param_group["lr"] * config["decay_rate"]
#                 param_group["lr"] = new_lr

#         epoch_loss = 0.0
#         epoch_acc = 0.0
#         epoch_loss_p_x = 0.0
#         epoch_loss_p_y_x = 0.0
#         epoch_loss_l2 = 0.0
#         loss_p_x = 0.0
#         loss_l2 = 0.0

#         """---TRAINING---"""

#         progress_bar = tqdm(
#             dload_train,
#             desc=(f"Epoch {epoch+1}/{config['n_epochs']}"),
#             disable=not accelerator.is_main_process if accelerator else False,
#         )

#         f.train()
#         for i, (x_p_d, _) in enumerate(progress_bar):
#             """Warmup Learning Rate"""
#             if cur_iter <= config["warmup_iters"]:
#                 lr = config["lr"] * cur_iter / float(config["warmup_iters"])
#                 for param_group in optim.param_groups:
#                     param_group["lr"] = lr

#             x_lab, y_lab = dload_train_labeled.__next__()
#             x_lab, y_lab = (x_lab.to(device), y_lab.to(device).squeeze().long())

#             L = 0.0

#             """Maximize log P(x)"""
#             if config["p_x_weight"] > 0:
#                 if accelerator is not None:
#                     with accelerator.no_sync(f):
#                         fp_all = f(x_p_d)
#                         fp = fp_all.mean()

#                         x_q = sample_q(f, datamodule, replay_buffer, accelerator=accelerator, **config)
#                         fq_all = f(x_q)
#                         fq = fq_all.mean()

#                         loss_p_x = fq - fp
#                         L += config["p_x_weight"] * loss_p_x

#                         if config["l2_weight"] > 0:
#                             loss_l2 = (fq**2 + fp**2).mean() * config["l2_weight"]
#                             L += loss_l2
#                 else:
#                     fp_all = f(x_p_d)
#                     fp = fp_all.mean()

#                     x_q = sample_q(f, datamodule, replay_buffer, accelerator=accelerator, **config)
#                     fq_all = f(x_q)
#                     fq = fq_all.mean()

#                     loss_p_x = fq - fp
#                     L += config["p_x_weight"] * loss_p_x

#                     if config["l2_weight"] > 0:
#                         loss_l2 = (fq**2 + fp**2).mean() * config["l2_weight"]
#                         L += loss_l2

#             """Maximize log P(y|x)"""
#             if config["p_y_x_weight"] > 0:
#                 logits = accelerator.unwrap_model(f).classify(x_lab) if accelerator else f.classify(x_lab)
#                 loss_p_y_x = nn.functional.cross_entropy(logits, y_lab)
#                 if logits.dim() > 1:
#                     acc = (logits.max(1)[1] == y_lab).float().mean()
#                 else:
#                     acc = (logits[0].argmax() == y_lab.item()).float()
#                 L += config["p_y_x_weight"] * loss_p_y_x

#             epoch_loss += L.item()
#             epoch_acc += acc.item()
#             epoch_loss_p_x += loss_p_x.item() if config["p_x_weight"] > 0 else 0.0
#             epoch_loss_l2 += loss_l2.item() if config["l2_weight"] > 0 else 0.0
#             epoch_loss_p_y_x += loss_p_y_x.item()

#             """Take gradient step"""
#             optim.zero_grad()
#             if accelerator:
#                 accelerator.backward(L)
#             else:
#                 L.backward()
#             optim.step()
#             cur_iter += 1

#         """---VALIDATION---"""

#         f.eval()
#         all_corrects, all_losses, all_confs, all_gts = [], [], [], []
#         val_loss, val_acc = np.inf, 0.0
#         for inputs, labels in dload_valid:
#             inputs, labels = inputs.to(device), labels.to(device).squeeze().long()

#             with t.no_grad():
#                 logits = accelerator.unwrap_model(f).classify(inputs) if accelerator else f.classify(inputs)

#             if accelerator:
#                 losses, corrects, confs, targets = accelerator.gather_for_metrics(
#                     (
#                         t.nn.functional.cross_entropy(logits, labels, reduction="none"),
#                         (logits.max(1)[1] == labels).float(),
#                         t.nn.functional.softmax(logits, dim=1),
#                         labels,
#                     )
#                 )
#             else:
#                 losses, corrects, confs, targets = (
#                     t.nn.functional.cross_entropy(logits, labels, reduction="none"),
#                     (logits.max(1)[1] == labels).float(),
#                     t.nn.functional.softmax(logits, dim=1),
#                     labels,
#                 )

#             all_gts.extend(targets)
#             all_confs.extend(confs)

#             all_losses.extend(loss.item() for loss in losses)
#             all_corrects.extend(correct.item() for correct in corrects)

#         all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape((-1, datamodule.n_classes))
#         all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

#         val_ece = ECE(10).measure(all_confs, all_gts)
#         val_loss = np.mean(all_losses)
#         val_acc = np.mean(all_corrects)

#         """Check if current valid loss is the best"""
#         if val_loss < best_val_loss:
#             best_val_loss, best_val_acc, best_val_ece = val_loss, val_acc, val_ece
#             print(f"BEST val_loss: {best_val_loss:.4f}", f"val_acc: {best_val_acc:.4f}", f"val_ece: {best_val_ece:.4f}", sep="\t")

#             if accelerator:
#                 if accelerator.is_main_process:
#                     if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
#                         os.remove(best_ckpt_path)

#                     best_ckpt_path = f"{ckpt_dir}/epoch={epoch + (config['n_epochs'] * iter_num)}-val_loss={val_loss:.4f}.ckpt"
#                     os.makedirs(ckpt_dir, exist_ok=True)

#                     ckpt_dict = {
#                         "model_state_dict": accelerator.unwrap_model(f).state_dict(),
#                         "optimizer_state_dict": optim.state_dict(),
#                         "replay_buffer": replay_buffer,
#                     }
#                     accelerator.save(ckpt_dict, best_ckpt_path)
#             else:
#                 if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
#                     os.remove(best_ckpt_path)

#                 best_ckpt_path = f"{ckpt_dir}/epoch={epoch + (config['n_epochs'] * iter_num)}-val_loss={val_loss:.4f}.ckpt"
#                 os.makedirs(ckpt_dir, exist_ok=True)

#                 ckpt_dict = {
#                     "model_state_dict": accelerator.unwrap_model(f).state_dict() if accelerator else f.state_dict(),
#                     "optimizer_state_dict": optim.state_dict(),
#                     "replay_buffer": replay_buffer,
#                 }
#                 t.save(ckpt_dict, best_ckpt_path)

#         """---LOGGING AND CHECKPOINTING---"""

#         if (epoch + (config["n_epochs"] * iter_num)) % config["sample_every_n_epochs"] == 0 and config["p_x_weight"] > 0:
#             x_q = sample_q(f, datamodule, replay_buffer, accelerator=accelerator, **config)

#             image = tv.utils.make_grid(x_q, normalize=True, nrow=8, value_range=(-1, 1))

#             if not os.path.exists(samples_dir):
#                 os.makedirs(samples_dir, exist_ok=True)

#             tv.utils.save_image(image, f"{samples_dir}/x_q-epoch={epoch + (config['n_epochs'] * iter_num)}.png")

#         epoch_loss /= len(dload_train)
#         epoch_acc /= len(dload_train)
#         epoch_loss_p_x /= len(dload_train)
#         epoch_loss_p_y_x /= len(dload_train)
#         epoch_loss_l2 /= len(dload_train)

#         print(
#             f"loss: {epoch_loss:.4f}",
#             f"acc: {epoch_acc:.4f}",
#             f"loss_p_x: {epoch_loss_p_x:.4f}",
#             f"loss_l2: {epoch_loss_l2:.4f}",
#             f"loss_p_y_x: {epoch_loss_p_y_x:.4f}",
#             f"val_loss: {val_loss:.4f}",
#             f"val_acc: {val_acc:.4f}",
#             f"val_ece: {val_ece:.4f}",
#             sep="\t",
#         )

#         if config["enable_tracking"]:
#             log_values = {
#                 "epoch": epoch + (config["n_epochs"] * iter_num) + 1,
#                 "loss": epoch_loss,
#                 "loss_p_x": epoch_loss_p_x,
#                 "loss_l2": epoch_loss_l2,
#                 "loss_p_y_x": epoch_loss_p_y_x,
#                 "acc": epoch_acc,
#                 "val_loss": val_loss,
#                 "val_acc": val_acc,
#                 "val_ece": val_ece,
#             }
#             log_fn(log_values)

#     """Log the final epoch"""
#     if config["enable_tracking"]:
#         log_fn(
#             {
#                 "num_labeled": len(datamodule.train_labeled_indices),
#                 "loss": epoch_loss,
#                 "loss_p_x": epoch_loss_p_x,
#                 "loss_l2": epoch_loss_l2,
#                 "loss_p_y_x": epoch_loss_p_y_x,
#                 "acc": epoch_acc,
#                 "val_loss": val_loss,
#                 "val_acc": val_acc,
#                 "val_ece": val_ece,
#             }
#         )

#     """Log the best epoch"""
#     if config["enable_tracking"]:
#         log_fn(
#             {
#                 "num_labeled": len(datamodule.train_labeled_indices),
#                 "best_val_loss": best_val_loss,
#                 "best_val_acc": best_val_acc,
#                 "best_val_ece": best_val_ece,
#             }
#         )

#     """Save the last checkpoint"""
#     ckpt_dict = {
#         "model_state_dict": accelerator.unwrap_model(f).state_dict() if accelerator else f.state_dict(),
#         "optimizer_state_dict": optim.state_dict(),
#         "replay_buffer": replay_buffer,
#     }

#     if accelerator:
#         if accelerator.is_main_process:
#             accelerator.save(ckpt_dict, f"{ckpt_dir}/last.ckpt")
#     else:
#         t.save(ckpt_dict, f"{ckpt_dir}/last.ckpt")

#     return f


def get_optim(model: nn.Module, args):
    if args.optim == "adam":
        optim = t.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        optim = t.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optim} not supported.")

    return optim


def get_model(datamodule: DataModule, args):
    model = F(args.depth, args.width, datamodule.n_classes, datamodule.img_shape[0])

    return model


def train_model(args):
    device = "cuda" if t.cuda.is_available() else "cpu"
    dm = DataModule(dataset=args.dataset, root_dir=args.root_dir, batch_size=args.batch_size, sigma=args.sigma)

    if not os.path.isfile(f"weights/{dm.dataset}_cov.pt"):
        category_mean(datamodule=dm)

    f = get_model(dm, args)
    replay_buffer = init_from_centers(datamodule=dm, buffer_size=args.buffer_size, device=device)
    optimizer = get_optim(f, args)

    trainer = CustomTrainer(f, optimizer, dm, replay_buffer, device, args)
    trainer.fit()
    # train_model(f=model, optim=optim, datamodule=dm, replay_buffer=replay_buffer, dirs=dirs, **config)


if __name__ == "__main__":
    t.backends.cudnn.enabled = True
    t.backends.cudnn.benchmark = True
    t.backends.cudnn.deterministic = False

    args = get_args()
    set_seed(args.seed)
    train_model(args)
