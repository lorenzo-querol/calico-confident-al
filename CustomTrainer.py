import os

import torch as t
import torch.nn as nn
from netcal.metrics import ECE
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from DataModule import DataModule
from Sampler import Sampler
from utils import disable_running_stats, enable_running_stats


class CustomTrainer:
    def __init__(self, f: nn.Module, opt: t.optim.Optimizer, datamodule: DataModule, device, args):
        self.f = f
        self.opt = opt
        self.datamodule = datamodule
        self.device = device
        self.hparams = args

        self.sampler = Sampler(
            datamodule=self.datamodule,
            buffer_size=self.hparams.buffer_size,
            n_steps=self.hparams.n_steps,
            sgld_lr=self.hparams.sgld_lr,
            reinit_freq=self.hparams.reinit_freq,
            sgld_std=self.hparams.sgld_std,
            device=self.device,
        )

        self.train_dataloader = self.datamodule.get_train_dataloader()
        self.labeled_dataloader = self.datamodule.get_labeled_dataloader()
        self.val_dataloader = self.datamodule.get_val_dataloader()

        self.f = self.f.to(self.device)
        self.ece = ECE(20)

        train_log_dir = f"./logs/{self.datamodule.dataset}/train"

        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)

        if not self.hparams.test:
            self.ver_num = len([name for name in os.listdir(train_log_dir)]) + 1
            self.trainer_writer = SummaryWriter(f"{train_log_dir}/v_{self.ver_num}")
        else:
            self.test_log_dir = f"./logs/{self.datamodule.dataset}/test"
            if not os.path.exists(self.test_log_dir):
                os.makedirs(self.test_log_dir)

            self.ver_num = self.hparams.ckpt_path.split("/")[-1].split("_")[-1]
            test_log_dir = f"./logs/{self.datamodule.dataset}/test"
            self.test_writer = SummaryWriter(f"{test_log_dir}/v_{self.ver_num}")

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

    def _calculate_px_loss(self, batch):
        x_p_d, _ = batch
        x_p_d = x_p_d.to(self.device)
        L = 0.0

        fp_all = self.f(x_p_d)
        fp = fp_all.mean()
        x_q = self.sampler.sample_q(self.f)
        fq_all = self.f(x_q)
        fq = fq_all.mean()

        cdiv_loss = -(fp - fq)
        L = self.hparams.px * cdiv_loss

        return L

    def _calculate_pyx_loss(self, batch):
        x_lab, y_lab = batch
        x_lab, y_lab = x_lab.to(self.device), y_lab.to(self.device).squeeze().long()
        L = 0.0

        logits = self.f.classify(x_lab)
        ce_loss = nn.functional.cross_entropy(logits, y_lab)
        acc = (logits.max(1)[1] == y_lab).float().mean()
        L += self.hparams.pyx * ce_loss

        return L, acc

    def _training_step(self, train_batch, labeled_batch):
        L, acc = 0.0, 0.0

        if self.hparams.px > 0:
            L += self._calculate_px_loss(train_batch)

        if self.hparams.pyx > 0:
            ce_loss, acc = self._calculate_pyx_loss(labeled_batch)
            L += ce_loss

        self.opt.zero_grad()
        L.backward()
        self.opt.step()

        return L, acc

    def _valid_step(self, batch):
        x_lab, y_lab = batch
        x_lab, y_lab = x_lab.to(self.device), y_lab.to(self.device).squeeze().long()

        with t.no_grad():
            logits = self.f.classify(x_lab)

        L = nn.functional.cross_entropy(logits, y_lab)
        acc = (logits.max(1)[1] == y_lab).float().mean()
        confs = t.nn.functional.softmax(logits, dim=1)

        return L, acc, confs, y_lab

    def _log_metrics(self, metrics: dict, mode: str):
        writer = self.trainer_writer if mode == "train" else self.test_writer

        # Log to tensorboard
        for key, value in metrics.items():
            writer.add_scalar(key, value, self.current_epoch)

        writer.flush()

        # Log to progress bar
        for key, value in metrics.items():
            metrics[key] = "{:.4f}".format(value)

        metrics["ver_num"] = self.ver_num

        self.progress_bar.set_postfix(metrics)

    def _compute_ece(self, confs, gts):
        confs, gts = t.cat(confs), t.cat(gts)
        all_confs = confs.cpu().numpy().reshape((-1, self.datamodule.n_classes))
        all_gts = gts.cpu().numpy()

        return self.ece.measure(all_confs, all_gts)

    def _check_best_valid(self, metrics):
        if metrics["val/loss"] < self.best_val_loss:
            ckpt_path = f"checkpoints/{self.datamodule.dataset}/train/v_{self.ver_num}"

            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)

            t.save(self.f.state_dict(), f"{ckpt_path}/best.pt")

    def _log_last_ckpt(self):
        ckpt_path = f"checkpoints/{self.datamodule.dataset}/train/v_{self.ver_num}"
        t.save(self.f.state_dict(), f"{ckpt_path}/last.pt")

    def _init_trackers(self):
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.current_iter = 0

    def _close_writers(self):
        self.trainer_writer.close()

        if hasattr(self, "test_writer") and self.test_writer is not None:
            self.test_writer.close()

    def fit(self):
        self._init_trackers()
        self.progress_bar = tqdm(range(self.hparams.n_epochs), desc=(f"Training Progress"), total=self.hparams.n_epochs, position=0, leave=True)

        for epoch in self.progress_bar:
            self.current_epoch = epoch
            self._decay_lr()

            batch_prog_bar = tqdm(
                zip(self.train_dataloader, self.labeled_dataloader),
                desc=f"Batch Progress",
                total=len(self.datamodule.full_train),
                position=1,
                leave=False,
            )

            # Train
            self.f.train()
            train_loss, train_acc = [], []
            for train_batch, labeled_batch in batch_prog_bar:
                self.current_iter += 1
                self._warmup_lr()

                loss, acc = self._training_step(train_batch, labeled_batch)
                train_loss.append(loss)
                train_acc.append(acc)

            train_loss = t.mean(t.stack(train_loss))
            train_acc = t.mean(t.stack(train_acc))

            # Validate
            self.f.eval()
            val_loss, val_acc, all_confs, all_gts = [], [], [], []
            for batch in self.val_dataloader:
                loss, acc, confs, gts = self._valid_step(batch)

                val_loss.append(loss)
                val_acc.append(acc)
                all_confs.append(confs)
                all_gts.append(gts)

            val_loss = t.mean(t.stack(val_loss))
            val_acc = t.mean(t.stack(val_acc))
            val_ece = self._compute_ece(all_confs, all_gts)

            metrics = {
                "train/loss": train_loss.item(),
                "train/acc": train_acc.item(),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item(),
                "val/ece": val_ece,
            }

            self._check_best_valid(metrics)
            self._log_metrics(metrics, "train")

        self._log_last_ckpt()
        self._close_writers()

    def test(self, ckpt_path: str | None, type: str | None = "best"):

        self.f.load_state_dict(t.load(f"{ckpt_path}/{type}.pt"))

        self.current_epoch = 0
        self.test_dataloader = self.datamodule.get_test_dataloader()
        self.progress_bar = tqdm(self.test_dataloader, desc=(f"Test Progress"), total=len(self.test_dataloader), position=0, leave=True)

        self.f.eval()
        test_loss, test_acc, all_confs, all_gts = [], [], [], []
        for batch in self.progress_bar:
            loss, acc, confs, gts = self._valid_step(batch)

            test_loss.append(loss)
            test_acc.append(acc)
            all_confs.append(confs)
            all_gts.append(gts)

        test_loss = t.mean(t.stack(test_loss))
        test_acc = t.mean(t.stack(test_acc))
        test_ece = self._compute_ece(all_confs, all_gts)

        metrics = {
            "test/loss": test_loss.item(),
            "test/acc": test_acc.item(),
            "test/ece": test_ece,
        }

        self._log_metrics(metrics, "test")
