from math import e
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as tr
import medmnist
import numpy as np
import torch as t
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from accelerate import Accelerator


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
        **config,
    ):
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
            common_transform = common_transform[:-1]

        if augment:
            final_transform = [tr.Pad(4, padding_mode="reflect"), tr.RandomCrop(32), tr.RandomHorizontalFlip(), tr.RandomVerticalFlip()]

            if self.dataset == "svhn":
                final_transform = final_transform[:-2]

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
            CIFAR10(root=self.data_root, transform=None, train=True if split == "train" else False, download=True)
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
            dataset = CIFAR10(root=self.data_root, transform=transform, train=train, download=False)

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

    def create_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last, pin_memory=True)

    def prepare_ddp(self):
        """Prepare dataloaders for Distributed Data Parallel (DDP)."""
        self.dload_train, self.dload_train_labeled, self.dload_train_unlabeled, self.dload_valid = self.accelerator.prepare(
            self.dload_train, self.dload_train_labeled, self.dload_train_unlabeled, self.dload_valid
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
        self.full_train = self._dataset_function("train", train=True, augment=False)

        self.all_train_indices = list(range(len(self.full_train)))
        self.train_labeled_indices = train_labeled_indices
        self.train_unlabeled_indices = train_unlabeled_indices

        """Semi-Supervised Learning"""
        train_indices = np.array(self.all_train_indices)
        train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

        if start_iter:
            if self.labels_per_class > 0 and sampling_method == None:
                """Equal number of samples per class"""
                self.train_labeled_indices = []
                self.train_unlabeled_indices = []

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

        self.accelerator.print(f"Current Labeled Train Indices: {str(len(self.train_labeled_indices))}")
        self.accelerator.print(f"Current Unlabeled Train Indices: {str(len(self.train_unlabeled_indices))}")

        self.labeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_labeled_indices)
        self.unlabeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_unlabeled_indices)
        self.valid = self._dataset_function("val", train=False, augment=False)

        self.dload_train = self.create_dataloader(self.full_train)
        self.dload_train_labeled = cycle(self.create_dataloader(self.labeled))
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

    def get_test_data(self):
        self.test = self._dataset_function("test", train=False, augment=False)
        self.dload_test = self.create_dataloader(self.test, shuffle=False, drop_last=False)

        if t.cuda.device_count() > 1:
            self.dload_test = self.accelerator.prepare(self.dload_test)

        return self.dload_test

    def query_samples(self, f: nn.Module, dload_train_unlabeled: DataLoader, train_unlabeled_inds: list[int], n_classes: int, query_size: int):
        confs, confs_to_fix = [], []

        f.eval()
        with t.no_grad():
            progress_bar = tqdm(dload_train_unlabeled, desc="Predicting Unlabeled", disable=not self.accelerator.is_main_process)
            for i, (x, y) in enumerate(progress_bar):
                x, y = x.to(self.accelerator.device), y.to(self.accelerator.device).squeeze().long()
                logits = self.accelerator.unwrap_model(f).classify(x)

                confs.extend(nn.functional.softmax(logits, 1).cpu().numpy().tolist())
                confs_to_fix = [(conf.max(), train_unlabeled_inds[i]) for i, conf in enumerate(np.array(confs).reshape((-1, n_classes)))]

                """Sort by confidence and take top query_size"""
                confs_to_fix.sort(key=lambda x: x[0])
                confs_to_fix = confs_to_fix[:query_size]

                inds_to_fix = [ind for conf, ind in confs_to_fix]
                inds_to_fix.sort()

            return inds_to_fix

    def random_sampling(self, train_unlabeled_inds: list[int], query_size: int):
        inds_to_fix = np.random.choice(train_unlabeled_inds, query_size, replace=False)
        inds_to_fix.sort()

        return inds_to_fix

    def get_class_distribution(self):
        class_labels = [self.classnames[self.full_train[idx][1][0]] for idx in self.train_labeled_indices]
        num_samples_added_per_class = defaultdict(int)

        for label in class_labels:
            num_samples_added_per_class[label] += 1

        counts = sorted(num_samples_added_per_class.items(), key=lambda x: x[0])

        return counts
