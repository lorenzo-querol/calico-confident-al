import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import medmnist
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
import yaml
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# from ExpUtils import AverageMeter


class Hamiltonian(_Loss):
    def __init__(self, layer, reg_cof=1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        y = self.layer(x)
        H = torch.sum(y * p)
        # H = H - self.reg_cof * l2
        return H


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as file:
        return yaml.safe_load(file)


def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpt-%04d.pth" % epoch)
    torch.save(state, filename)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(args, bs, im_sz=32, n_ch=3):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


class DataModule:
    def __init__(self, accelerator, sigma, batch_size, labels_per_class, data_root, dataset, **config):
        self.accelerator = accelerator
        self.sigma = sigma
        self.batch_size = batch_size
        self.labels_per_class = labels_per_class
        self.data_root = data_root
        self.dataset = dataset

    def get_transforms(self, train: bool, augment: bool):
        final_transform = []
        common_transform = [
            tr.ToTensor(),
            tr.Normalize((0.5,) * self.img_shape[0], (0.5,) * self.img_shape[0]),
        ]

        if self.img_shape[0] == 1:
            final_transform.pop()

        if augment:
            final_transform = [tr.Pad(4, padding_mode="reflect"), tr.RandomCrop(32), tr.RandomHorizontalFlip(), tr.RandomVerticalFlip()]

            if self.dataset == "svhn":
                final_transform.pop()
                final_transform.pop()

        final_transform.extend(common_transform)

        if train:
            final_transform.extend([lambda x: x + self.sigma * t.randn_like(x)])

        return tr.Compose(final_transform)

    def prepare_data(self):
        if self.accelerator.is_main_process:
            self.download_dataset("train")
            self.download_dataset("val")
            self.download_dataset("test")

    def download_dataset(self, split: str):
        if self.dataset == "cifar10":
            tv.datasets.CIFAR10(root=self.data_root, transform=None, train=True if split == "train" else False, download=True)
        elif self.dataset in ["bloodmnist", "organcmnist", "organamnist", "organsmnist", "dermamnist", "pneumoniamnist"]:
            info = medmnist.INFO[self.dataset]
            DataClass = getattr(medmnist, info["python_class"])
            DataClass(root=self.data_root, transform=None, split=split, download=True)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

    def _dataset_function(self, split: str, train: bool, augment: bool):
        if self.dataset == "cifar10":
            self.img_shape = (3, 32, 32)
            self.n_classes = 10

            transform = self.get_transforms(train=train, augment=augment)
            dataset = tv.datasets.CIFAR10(root=self.data_root, transform=transform, train=train, download=False)

            self.classnames = dataset.classes

            return dataset

        elif self.dataset in ["bloodmnist", "organcmnist", "organamnist", "organsmnist", "dermamnist", "pneumoniamnist"]:
            info = medmnist.INFO[self.dataset]
            DataClass = getattr(medmnist, info["python_class"])
            classnames = info["label"]

            self.img_shape = (info["n_channels"], 28, 28)

            transform = self.get_transforms(train=train, augment=augment)
            dataset = DataClass(split=split, transform=transform, download=False)

            self.classnames = [classnames[str(i)] for i in range(len(classnames))]
            self.n_classes = len(classnames)

            return dataset

    def create_dataloader(self, dataset, shuffle=True, drop_last=True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last, pin_memory=True)

    def get_data(
        self, override_labels_per_class=None, train_labeled_indices=None, train_unlabeled_indices=None, indices_to_fix=None, start_iter=True
    ):
        self.full_train = self._dataset_function("train", train=True, augment=False)

        self.all_train_indices = list(range(len(self.full_train)))
        self.train_labeled_indices = train_labeled_indices
        self.train_unlabeled_indices = train_unlabeled_indices

        # Semi-Supervision
        train_indices = np.array(self.all_train_indices)
        train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

        if start_iter:
            if self.labels_per_class > 0 or override_labels_per_class is not None:
                labels_per_class = self.labels_per_class if override_labels_per_class is None else override_labels_per_class

                self.train_labeled_indices = []
                self.train_unlabeled_indices = []

                for i in range(self.n_classes):
                    self.train_labeled_indices.extend(train_indices[train_labels == i][:labels_per_class])
                    self.train_unlabeled_indices.extend(train_indices[train_labels == i][labels_per_class:])
            else:
                self.train_labeled_indices = train_indices
                self.train_unlabeled_indices = []
        else:
            self.train_labeled_indices = np.append(self.train_labeled_indices, indices_to_fix)
            indices = np.argwhere(np.isin(self.train_unlabeled_indices, indices_to_fix))
            self.train_unlabeled_indices = np.delete(self.train_unlabeled_indices, indices)

        self.accelerator.print(f"Current Labeled Train Indices: {str(len(self.train_labeled_indices))}")
        self.accelerator.print(f"Current Unlabeled Train Indices: {str(len(self.train_unlabeled_indices))}")

        self.labeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_labeled_indices)
        self.unlabeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_unlabeled_indices)
        self.valid = self._dataset_function("val", train=False, augment=False)

        self.dload_train = self.create_dataloader(self.full_train)
        self.dload_train_labeled = cycle(self.create_dataloader(self.labeled))
        self.dload_train_unlabeled = self.create_dataloader(self.unlabeled) if len(self.train_unlabeled_indices) > 0 else None
        self.dload_valid = self.create_dataloader(self.valid, shuffle=False, drop_last=False)

        self.dload_train, self.dload_train_labeled, self.dload_train_unlabeled, self.dload_valid = self.accelerator.prepare(
            self.dload_train, self.dload_train_labeled, self.dload_train_unlabeled, self.dload_valid
        )

        return (
            self.dload_train,
            self.dload_train_labeled,
            self.dload_train_unlabeled,
            self.dload_valid,
            self.train_labeled_indices,
            self.train_unlabeled_indices,
        )

    def get_test_data(self):
        self.test = self._dataset_function("test", train=False, augment=False)
        self.dload_test = self.create_dataloader(self.test, shuffle=False, drop_last=False)

        return self.dload_test

    def query_samples(self, f, dload_unlabeled, train_unlabeled_inds, n_classes, query_size):
        confs, confs_to_fix = [], []

        progress_bar = tqdm(dload_unlabeled, desc="Predicting Unlabeled", disable=not self.accelerator.is_main_process)
        for i, (x_p_d, y_p_d) in enumerate(progress_bar):
            x_p_d, y_p_d = x_p_d.to(self.accelerator.device), y_p_d.to(self.accelerator.device).squeeze().long()
            logits = self.accelerator.unwrap_model(f).classify(x_p_d)

            with t.no_grad():
                confs.extend(nn.functional.softmax(logits, 1).cpu().numpy())

        with t.no_grad():
            confs = np.array(confs).reshape((-1, n_classes))

            for ind, conf in enumerate(confs):
                confs_to_fix.append((conf.max(), train_unlabeled_inds[ind]))

            confs_to_fix.sort(key=lambda x: x[0])  # Sorts by confidence for each image

            confs_to_fix = confs_to_fix[:query_size]
            inds_to_fix = [ind for conf, ind in confs_to_fix]
            inds_to_fix.sort()

            return inds_to_fix

    def get_class_dist(self):
        class_labels = [self.classnames[self.full_train[idx][1][0]] for idx in self.train_labeled_indices]
        num_samples_added_per_class = defaultdict(int)

        for label in class_labels:
            num_samples_added_per_class[label] += 1

        counts = sorted(num_samples_added_per_class.items(), key=lambda x: x[0])

        return counts


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_classification(f, dload, set_name, epoch, args=None, print=None):
    corrects, losses = [], []

    for x, y in dload:
        x, y = x.to(args.device), y.to(args.device)
        logits = f.module.classify(x)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y).detach().cpu().numpy()
        correct = (logits.max(1)[1] == y).float().cpu().numpy()

        losses.extend(loss)
        corrects.extend(correct)

    loss = np.mean(losses)

    return correct, loss
