import os

import medmnist
import numpy as np
import pandas as pd
import torch as t
import torchvision.transforms as tr
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CustomDataset import CustomDataset

BENCHMARK_DATASETS = ["mnist", "cifar10", "cifar100", "svhn"]
MEDMNIST_DATASETS = ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]


class DataSubset(Dataset):
    def __init__(self, base_dataset, indices=None, size=-1):
        self.base_dataset = base_dataset
        if indices is None:
            indices = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.indices = indices

    def __getitem__(self, index):
        base_idx = self.indices[index]
        return self.base_dataset[base_idx]

    def __len__(self):
        return len(self.indices)


class DataModule:
    def __init__(self, dataset: str, root_dir: str = "./data", batch_size: int = 64, sigma: float = 3e-2):
        self.sigma = sigma
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.dataset = dataset

        self._prepare_data()

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
        if self.dataset in BENCHMARK_DATASETS:
            other_dataset = CustomDataset(self.dataset, root=self.root_dir, split=split, transform=None, download=True)
            self.img_shape = (1, 28, 28) if self.dataset == "mnist" else (3, 32, 32)
            self.n_classes = 100 if self.dataset == "cifar100" else 10
            self.classes = other_dataset.classes

        elif self.dataset in MEDMNIST_DATASETS:
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

        if self.dataset in BENCHMARK_DATASETS:
            other_dataset = CustomDataset(self.dataset, root=self.root_dir, split=split, transform=transform, download=False)
            dataset = other_dataset.get_dataset()

            return dataset

        elif self.dataset in MEDMNIST_DATASETS:
            info = medmnist.INFO[self.dataset]
            DataClass = getattr(medmnist, info["python_class"])
            dataset = DataClass(root=self.root_dir, split=split, transform=transform, download=False)

            return dataset

        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

    def setup(
        self,
        indices_to_fix: list = None,
        sample_method: str = None,
        init_size: int = None,
        labels_per_class: int = None,
        start_iter: bool = True,
        log_dir: str = None,
    ):
        self.full_train = self._dataset_function("train", train=True, augment=False)
        self.all_train_indices = list(range(len(self.full_train)))

        # Semi-Supervised Learning (SSL)
        train_indices = np.array(self.all_train_indices)
        self.train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

        if start_iter:
            # Equal number of labels per class
            if sample_method == "equal":
                assert labels_per_class is not None, "labels_per_class must be specified for equal sampling."

                self.labeled_indices = []
                self.unlabeled_indices = []

                for i in range(self.n_classes):
                    self.labeled_indices.extend(train_indices[self.train_labels == i][:labels_per_class])
                    self.unlabeled_indices.extend(train_indices[self.train_labels == i][labels_per_class:])

            # Random sampling
            elif sample_method == "random":
                assert init_size is not None, "init_size must be specified for random sampling."

                # set seed
                np.random.seed(1)

                self.labeled_indices = np.random.choice(train_indices, init_size, replace=False)
                self.unlabeled_indices = np.setdiff1d(train_indices, self.labeled_indices)

            # Use all training data
            else:
                self.labeled_indices = train_indices
                self.unlabeled_indices = []
        else:
            self.labeled_indices = np.append(self.labeled_indices, indices_to_fix)
            indices = np.argwhere(np.isin(self.unlabeled_indices, indices_to_fix))
            self.unlabeled_indices = np.delete(self.unlabeled_indices, indices)

        print("Sampling method:", sample_method)
        print("Current labeled train indices:", len(self.labeled_indices))
        print("Current unlabeled train indices:", len(self.unlabeled_indices))

        self.labeled = DataSubset(self._dataset_function("train", train=True, augment=True), indices=self.labeled_indices)
        self.labeled_labels = np.array([np.squeeze(self.labeled[ind][1]) for ind in range(len(self.labeled))])

        self.unlabeled = DataSubset(self._dataset_function("train", train=False, augment=False), indices=self.unlabeled_indices)

        self.val = self._dataset_function("val", train=False, augment=False)
        self.test = self._dataset_function("test", train=False, augment=False)

        # Calculate class distribution
        labels, counts = np.unique(self.labeled_labels, return_counts=True)
        distribution_dict = {}
        distribution_dict["num_labeled"] = [len(self.labeled_indices)]

        for label, count in zip(labels, counts):
            distribution_dict[label] = [count]

        print("Class Distribution:")
        for key, value in distribution_dict.items():
            if key == "num_labeled":
                continue
            print(f"Class {key}: {value[0]}")

        distribution_df = pd.DataFrame(distribution_dict)
        distribution_df.columns = ["num_labeled"] + self.classes

        if os.path.exists(f"{log_dir}/class_dist.csv"):
            distribution_df.to_csv(f"{log_dir}/class_dist.csv", mode="a", header=False, index=False)
        else:
            distribution_df.to_csv(f"{log_dir}/class_dist.csv", mode="w", header=True, index=False)

    def test_setup(self, test_dir: str):
        self.test = self._dataset_function("test", train=False, augment=False)
        self.test_indices = list(range(len(self.test)))
        self.test_labels = np.array([np.squeeze(self.test[ind][1]) for ind in self.test_indices])

        # Calculate class distribution
        labels, counts = np.unique(self.test_labels, return_counts=True)
        distribution_dict = {}
        distribution_dict["num_labeled"] = [len(self.test_indices)]

        for label, count in zip(labels, counts):
            distribution_dict[label] = [count]

        print("Class Distribution:")
        for key, value in distribution_dict.items():
            if key == "num_labeled":
                continue
            print(f"Class {key}: {value[0]}")

        distribution_df = pd.DataFrame(distribution_dict)
        distribution_df.columns = ["num_labeled"] + self.classes

        distribution_df.to_csv(f"{test_dir}/class_dist.csv", mode="w", header=True, index=False)

    def query(self, f: t.nn.Module, query_size: int, log_dir: str):
        progress_bar = tqdm(self.unlabeled_dataloader(), desc="Prediction Progress")
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        f.to(device)

        confs, confs_to_fix = [], []
        y_true, y_pred = [], []

        f.eval()
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device).squeeze().long()

            with t.no_grad():
                logits = f.classify(x)

            confs.extend(t.nn.functional.softmax(logits, 1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        confs = np.array(confs).reshape((-1, self.n_classes))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes)))
        accuracies = cm.diagonal() / cm.sum(axis=1)
        accuracies_df = pd.DataFrame(accuracies).T

        num_labeled_df = pd.DataFrame([len(self.labeled_indices)], columns=["num_labeled"])
        accuracies_df = pd.concat([num_labeled_df, accuracies_df], axis=1)

        accuracies_df.columns = ["num_labeled"] + self.classes

        if os.path.exists(f"{log_dir}/acc_per_class.csv"):
            accuracies_df.to_csv(f"{log_dir}/acc_per_class.csv", mode="a", header=False, index=False)
        else:
            accuracies_df.to_csv(f"{log_dir}/acc_per_class.csv", mode="w", header=True, index=False)

        for idx, conf in enumerate(confs):
            confs_to_fix.append((conf.max(), self.unlabeled_indices[idx]))

        # Sorts by confidence for each image
        confs_to_fix.sort(key=lambda x: x[0])

        confs_to_fix = confs_to_fix[:query_size]
        indices_to_fix = [ind for conf, ind in confs_to_fix]
        indices_to_fix.sort()

        print(f"Indices to fix: {len(indices_to_fix)}")

        return indices_to_fix

    def _cycle(self, loader):
        while True:
            for data in loader:
                yield data

    def _create_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last, pin_memory=True)

    def train_dataloader(self):
        return self._create_dataloader(self.full_train, drop_last=True)

    def labeled_dataloader(self):
        return self._cycle(self._create_dataloader(self.labeled, drop_last=True))

    def unlabeled_dataloader(self):
        return self._create_dataloader(self.unlabeled, shuffle=False)

    def val_dataloader(self):
        return self._create_dataloader(self.val, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test, shuffle=False)
