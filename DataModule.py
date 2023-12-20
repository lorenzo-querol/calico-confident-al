import os
from collections import defaultdict
from math import e

import medmnist
import numpy as np
import torch as t
import torch.nn as nn
import torchvision.transforms as tr
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from tqdm import tqdm
from OtherDataset import OtherDataset


def cycle(loader):
    while True:
        for data in loader:
            yield data


class DataModule:
    def __init__(
        self,
        accelerator: Accelerator,
        sigma: float,
        batch_size: int,
        labels_per_class: int,
        data_root: str,
        dataset: str,
        query_size: int,
        **config,
    ):
        self.accelerator = accelerator
        self.sigma = sigma
        self.batch_size = batch_size
        self.labels_per_class = labels_per_class
        self.data_root = data_root
        self.dataset = dataset
        self.query_size = query_size

    def get_transforms(self, train: bool, augment: bool):
        final_transform = []
        common_transform = [
            tr.ToTensor(),
            tr.Normalize((0.5,) * self.img_shape[0], (0.5,) * self.img_shape[0]),
        ]

        if augment:
            final_transform = [
                tr.Pad(4, padding_mode="reflect"),
                tr.RandomCrop(self.img_shape[1]),
                tr.RandomRotation(20),
                tr.RandomHorizontalFlip(),
            ]

            if self.dataset == "svhn":
                final_transform = final_transform[:-1]

        final_transform.extend(common_transform)

        if train:
            final_transform.extend([lambda x: x + self.sigma * t.randn_like(x)])

        return tr.Compose(final_transform)

    def prepare_data(self):
        with self.accelerator.main_process_first():
            if not os.path.exists(self.data_root):
                os.makedirs(self.data_root)

            self.download_dataset("train")
            self.download_dataset("val")
            self.download_dataset("test")

    def download_dataset(self, split: str):
        with self.accelerator.main_process_first():
            if self.dataset == "mnist":
                MNIST(root=self.data_root, transform=None, train=True if split == "train" else False, download=True)
            elif self.dataset == "svhn":
                SVHN(root=self.data_root, transform=None, split=split, download=True)
            elif self.dataset == "cifar10":
                CIFAR10(root=self.data_root, transform=None, train=True if split == "train" else False, download=True)
            elif self.dataset == "cifar100":
                CIFAR100(root=self.data_root, transform=None, train=True if split == "train" else False, download=True)
            elif self.dataset in ["bloodmnist", "organcmnist", "dermamnist", "pneumoniamnist"]:
                info = medmnist.INFO[self.dataset]
                DataClass = getattr(medmnist, info["python_class"])
                DataClass(root=self.data_root, transform=None, split=split, download=True)
            else:
                raise ValueError(f"Dataset {self.dataset} not supported.")

    def _dataset_function(self, split: str, train: bool, augment: bool):
        if self.dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
            self.img_shape = (3, 32, 32)
            self.n_classes = 100 if self.dataset == "cifar100" else 10
            transform = self.get_transforms(train=train, augment=augment)

            other_dataset = OtherDataset(self.dataset, root=self.data_root, split=split, transform=transform, download=False)
            dataset = other_dataset.get_dataset()

            self.classnames = other_dataset.classes

            return dataset

        elif self.dataset in ["bloodmnist", "organcmnist", "dermamnist", "pneumoniamnist"]:
            info = medmnist.INFO[self.dataset]
            DataClass = getattr(medmnist, info["python_class"])
            classnames = info["label"]

            self.img_shape = (info["n_channels"], 28, 28)

            transform = self.get_transforms(train=train, augment=augment)
            dataset = DataClass(root=self.data_root, split=split, transform=transform, download=False)

            self.classnames = [classnames[str(i)] for i in range(len(classnames))]
            self.n_classes = len(classnames)

            return dataset

    def prepare_ddp(self):
        """Prepare dataloaders for Distributed Data Parallel (DDP)."""
        (
            self.dload_train,
            self.dload_train_labeled,
            self.dload_train_unlabeled,
            self.dload_valid,
        ) = self.accelerator.prepare(
            self.dload_train,
            self.dload_train_labeled,
            self.dload_train_unlabeled,
            self.dload_valid,
        )

    def get_data(
        self,
        train_labeled_indices: list = None,
        train_unlabeled_indices: list = None,
        indices_to_fix: list = None,
        sampling_method: str = None,
        init_size: int = None,
        start_iter: bool = True,
    ):
        self.train_labeled_indices = train_labeled_indices
        self.train_unlabeled_indices = train_unlabeled_indices
        valid_indices = None

        self.full_train = self._dataset_function("train", train=True, augment=False)
        self.all_train_indices = list(range(len(self.full_train)))

        """Semi-Supervised Learning"""
        train_indices = np.array(self.all_train_indices)
        train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

        if start_iter:
            if self.labels_per_class > 0 and sampling_method == None:
                for i in range(self.n_classes):
                    self.train_labeled_indices.extend(train_indices[train_labels == i][: self.labels_per_class])
                    self.train_unlabeled_indices.extend(train_indices[train_labels == i][self.labels_per_class :])

            elif sampling_method == "random" and init_size > 0:
                """Random sampling"""
                init_size = min(init_size, len(train_indices))
                self.train_labeled_indices = np.random.choice(train_indices, init_size, replace=False)
                self.train_unlabeled_indices = np.setdiff1d(train_indices, self.train_labeled_indices)

            else:
                """Use all training data"""
                self.train_labeled_indices = train_indices
                self.train_unlabeled_indices = []
        else:
            if sampling_method == "random":
                """Random sampling"""
                init_size = min(init_size, len(train_indices))
                self.train_labeled_indices = np.random.choice(train_indices, init_size, replace=False)
                self.train_unlabeled_indices = np.setdiff1d(train_indices, self.train_labeled_indices)
            else:
                self.train_labeled_indices = np.append(self.train_labeled_indices, indices_to_fix)
                indices = np.argwhere(np.isin(self.train_unlabeled_indices, indices_to_fix))
                self.train_unlabeled_indices = np.delete(self.train_unlabeled_indices, indices)

        self.accelerator.print(f"Current Labeled Train Indices: {len(self.train_labeled_indices)}")
        self.accelerator.print(f"Current Unlabeled Train Indices: {len(self.train_unlabeled_indices)}")

        # NOTE: Problem is this is with reference to the full train, and not the altered version with 20% off the training data
        self.labeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_labeled_indices)
        self.unlabeled = Subset(self._dataset_function("train", train=False, augment=False), indices=self.train_unlabeled_indices)
        self.valid = self._dataset_function("val", train=False, augment=False)

        self.dload_train = self.create_dataloader(self.full_train, train=True)
        self.dload_train_labeled = cycle(self.create_dataloader(self.labeled, drop_last=False, train=True))
        self.dload_train_unlabeled = self.create_dataloader(self.unlabeled) if len(self.train_unlabeled_indices) > 0 else None
        self.dload_valid = self.create_dataloader(self.valid, shuffle=False, drop_last=False)

        if t.cuda.device_count() > 1:
            self.prepare_ddp()

        return (
            self.dload_train,
            self.dload_train_labeled,
            self.dload_train_unlabeled,
            self.dload_valid,
            self.train_labeled_indices,
            self.train_unlabeled_indices,
        )

    def create_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = True, train: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size if train else self.query_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=drop_last,
            pin_memory=True,
        )

    def get_test_data(self):
        self.test = self._dataset_function("test", train=False, augment=False)
        self.dload_test = self.create_dataloader(self.test, shuffle=True, drop_last=False)

        return self.dload_test

    def query_samples(self, f: nn.Module, dload_train_unlabeled: DataLoader, train_unlabeled_inds: list[int], query_size: int):
        confs, confs_to_fix = [], []

        f.eval()
        progress_bar = tqdm(dload_train_unlabeled, disable=not self.accelerator.is_main_process)
        with t.no_grad():
            for i, (x_p_d, y_p_d) in enumerate(progress_bar):
                x_p_d, y_p_d = x_p_d.to(self.accelerator.device), y_p_d.to(self.accelerator.device).squeeze().long()
                logits = self.accelerator.unwrap_model(f).classify(x_p_d)

                confs.extend(nn.functional.softmax(logits, dim=1).cpu().numpy())

            confs = np.array(confs).reshape((-1, self.n_classes))

            for ind, conf in enumerate(confs):
                if ind >= len(train_unlabeled_inds):
                    break
                confs_to_fix.append((conf.max(), train_unlabeled_inds[ind]))

            """Sorts by confidence for each image"""
            confs_to_fix.sort(key=lambda x: x[0])

            # Ensure that the number of samples to be queried is not greater than the size of the unlabeled pool
            query_size = min(query_size, len(train_unlabeled_inds))

            confs_to_fix = confs_to_fix[:query_size]
            inds_to_fix = [ind for _, ind in confs_to_fix]
            inds_to_fix.sort()

            self.accelerator.print(f"Length of inds to fix: {len(inds_to_fix)}")

        return inds_to_fix

    def get_class_distribution(self):
        class_labels = [self.classnames[self.full_train[idx][1][0]] for idx in self.train_labeled_indices]
        num_samples_added_per_class = defaultdict(int)

        for label in class_labels:
            num_samples_added_per_class[label] += 1

        counts = sorted(num_samples_added_per_class.items(), key=lambda x: x[0])

        return counts
