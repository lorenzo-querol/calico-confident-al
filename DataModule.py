import os

import medmnist
import torch as t
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Subset

from CustomDataset import CustomDataset


class DataModule:
    def __init__(self, dataset: str, root_dir: str = "./data", batch_size: int = 64, sigma: float = 3e-2):
        self.sigma = sigma
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.dataset = dataset

        self._prepare_data()
        self._setup()

    def _get_transforms(self, train: bool, augment: bool):
        train_transform = []
        test_transform = [
            tr.ToTensor(),
            tr.Normalize((0.5,) * self.img_shape[0], (0.5,) * self.img_shape[0]),
        ]

        if augment:
            train_transform = [
                tr.Pad(4, padding_mode="reflect"),
                tr.RandomCrop(self.img_shape[1]),
                tr.RandomHorizontalFlip(),
            ]

            if self.dataset == "svhn":
                train_transform = train_transform[:-1]

        train_transform.extend(test_transform)

        if train:
            train_transform.extend([lambda x: x + self.sigma * t.randn_like(x)])

        return tr.Compose(train_transform)

    def _prepare_data(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self._download_dataset("train")
        self._download_dataset("val")
        self._download_dataset("test")

    def _download_dataset(self, split: str):
        if self.dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
            other_dataset = CustomDataset(self.dataset, root=self.root_dir, split=split, transform=None, download=True)
            self.img_shape = (1, 28, 28) if self.dataset == "mnist" else (3, 32, 32)
            self.classes = 100 if self.dataset == "cifar100" else 10
            self.classnames = other_dataset.classes

        elif self.dataset in ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]:
            info = medmnist.INFO[self.dataset]
            DataClass = getattr(medmnist, info["python_class"])
            classnames = info["label"]

            DataClass(root=self.root_dir, transform=None, split=split, download=True)
            self.img_shape = (info["n_channels"], 28, 28)
            self.classes = [classnames[str(i)] for i in range(len(classnames))]
            self.n_classes = len(classnames)

        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

    def _dataset_function(self, split: str, train: bool, augment: bool):
        transform = self._get_transforms(train=train, augment=augment)

        if self.dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
            other_dataset = CustomDataset(self.dataset, root=self.root_dir, split=split, transform=transform, download=False)
            dataset = other_dataset.get_dataset()

            return dataset

        elif self.dataset in ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]:
            info = medmnist.INFO[self.dataset]
            DataClass = getattr(medmnist, info["python_class"])
            dataset = DataClass(root=self.root_dir, split=split, transform=transform, download=False)

            return dataset

        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

    def _setup(self):
        self.train = self._dataset_function("train", train=True, augment=False)
        self.train_indices = t.arange(len(self.train))

        self.labeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_indices)
        self.val = self._dataset_function("val", train=False, augment=False)
        self.test = self._dataset_function("test", train=False, augment=False)

    def _cycle(self, loader):
        while True:
            for data in loader:
                yield data

    def _create_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last, pin_memory=True)

    def get_train_dataloader(self):
        self.full_train = self._create_dataloader(self.train, shuffle=True, drop_last=True)
        return self._cycle(self.full_train)

    def get_labeled_dataloader(self):
        return self._create_dataloader(self.labeled, drop_last=True)

    def get_val_dataloader(self):
        return self._create_dataloader(self.val, shuffle=False)

    def get_test_dataloader(self):
        return self._create_dataloader(self.test, shuffle=False)
