import os

import medmnist
import numpy as np
import pandas as pd
import torch as t
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import log_class_dist


DATASETS = ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]


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


class MedMNISTDataModule:
    def __init__(self, dataset: str, root_dir: str, batch_size: int, sigma: float, seed: int):
        self.sigma = sigma
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.dataset = dataset
        self.seed = seed

        self._prepare_data()

    def _get_transforms(self, train: bool, augment: bool):
        transforms = []
        test_transform = [tr.ToTensor(), tr.Normalize((0.5,) * self.img_shape[0], (0.5,) * self.img_shape[0])]

        if augment:
            transforms = [tr.Pad(4, padding_mode="reflect"), tr.RandomCrop(self.img_shape[1]), tr.RandomHorizontalFlip()]

        transforms.extend(test_transform)

        if train:
            transforms.extend([lambda x: x + self.sigma * t.randn_like(x)])

        return tr.Compose(transforms)

    def _prepare_data(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self._download_dataset("train")
        self._download_dataset("val")
        self._download_dataset("test")

    def _download_dataset(self, split: str):
        if self.dataset in DATASETS:
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

        if self.dataset in DATASETS:
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
        log_dir: str = None,
        start_iter: bool = True,
    ):
        self.full_train = self._dataset_function("train", train=True, augment=False)
        self.all_train_indices = list(range(len(self.full_train)))

        train_indices = np.array(self.all_train_indices)
        self.train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

        if start_iter:
            if sample_method == "equal":
                assert labels_per_class is not None, "argument 'labels_per_class' must be specified if 'sample_method' == equal"

                self.labeled_indices = []
                self.unlabeled_indices = []

                for i in range(self.n_classes):
                    self.labeled_indices.extend(train_indices[self.train_labels == i][:labels_per_class])
                    self.unlabeled_indices.extend(train_indices[self.train_labels == i][labels_per_class:])

            elif sample_method == "random":
                assert init_size is not None, "argument 'init_size' must be specified if 'sample_method' == random"

                np.random.seed(self.seed)

                self.labeled_indices = np.random.choice(train_indices, init_size, replace=False)
                self.unlabeled_indices = np.setdiff1d(train_indices, self.labeled_indices)

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

        log_class_dist(self.labeled_labels, self.labeled_indices, self.classes, log_dir)

    def test_setup(self, test_dir: str):
        self.val = self._dataset_function("val", train=False, augment=False)
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

    def query(self, f: t.nn.Module, query_size: int):
        progress_bar = tqdm(self.unlabeled_dataloader(), desc="Predicting")
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

        for idx, conf in enumerate(confs):
            confs_to_fix.append((conf.max(), self.unlabeled_indices[idx]))

        confs_to_fix.sort(key=lambda x: x[0])  # Sorts by confidence for each image

        confs_to_fix = confs_to_fix[:query_size]
        indices_to_fix = [ind for conf, ind in confs_to_fix]
        indices_to_fix.sort()

        print(f"Length of 'indices_to_fix': {len(indices_to_fix)}")

        return indices_to_fix

    def _cycle(self, loader):
        while True:
            for data in loader:
                yield data

    def _create_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last, pin_memory=True)

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
