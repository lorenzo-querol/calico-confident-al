# import os
# from collections import defaultdict
# from math import e

# import medmnist
# import numpy as np
# import torch as t
# import torch.nn as nn
# import torchvision.transforms as tr
# from accelerate import Accelerator
# from colorama import init
# from torch.utils.data import DataLoader, Subset

# from OtherDataset import OtherDataset


# def cycle(loader):
#     while True:
#         for data in loader:
#             yield data


# class DataModule:
#     def __init__(
#         self,
#         sigma: float,
#         batch_size: int,
#         labels_per_class: int,
#         data_root: str,
#         dataset: str,
#         query_size: int,
#         accelerator: Accelerator = None,
#         **config,
#     ):
#         self.accelerator = accelerator
#         self.sigma = sigma
#         self.batch_size = batch_size
#         self.labels_per_class = labels_per_class
#         self.data_root = data_root
#         self.dataset = dataset
#         self.query_size = query_size

#     def get_transforms(self, train: bool, augment: bool):
#         final_transform = []
#         common_transform = [
#             tr.ToTensor(),
#             tr.Normalize((0.5,) * self.img_shape[0], (0.5,) * self.img_shape[0]),
#         ]

#         if augment:
#             final_transform = [
#                 tr.Pad(4, padding_mode="reflect"),
#                 tr.RandomCrop(self.img_shape[1]),
#                 tr.RandomHorizontalFlip(),
#             ]

#             if self.dataset == "svhn":
#                 final_transform = final_transform[:-1]

#         final_transform.extend(common_transform)

#         if train:
#             final_transform.extend([lambda x: x + self.sigma * t.randn_like(x)])

#         return tr.Compose(final_transform)

#     def prepare_data(self):
#         if self.accelerator is None:
#             if not os.path.exists(self.data_root):
#                 os.makedirs(self.data_root)

#             self.download_dataset("train")
#             self.download_dataset("val")
#             self.download_dataset("test")

#             return

#         with self.accelerator.main_process_first():
#             if not os.path.exists(self.data_root):
#                 os.makedirs(self.data_root)

#             self.download_dataset("train")
#             self.download_dataset("val")
#             self.download_dataset("test")

#     def download_dataset(self, split: str):
#         if self.dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
#             other_dataset = OtherDataset(
#                 self.dataset,
#                 root=self.data_root,
#                 split=split,
#                 transform=None,
#                 download=True,
#             )
#             self.img_shape = (1, 28, 28) if self.dataset == "mnist" else (3, 32, 32)
#             self.n_classes = 100 if self.dataset == "cifar100" else 10
#             self.classnames = other_dataset.classes

#         elif self.dataset in ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]:
#             info = medmnist.INFO[self.dataset]
#             DataClass = getattr(medmnist, info["python_class"])
#             classnames = info["label"]

#             DataClass(root=self.data_root, transform=None, split=split, download=True)
#             self.img_shape = (info["n_channels"], 28, 28)
#             self.classnames = [classnames[str(i)] for i in range(len(classnames))]
#             self.n_classes = len(classnames)

#         else:
#             raise ValueError(f"Dataset {self.dataset} not supported.")

#     def _dataset_function(self, split: str, train: bool, augment: bool):
#         transform = self.get_transforms(train=train, augment=augment)

#         if self.dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
#             other_dataset = OtherDataset(self.dataset, root=self.data_root, split=split, transform=transform, download=False)
#             dataset = other_dataset.get_dataset()

#             return dataset

#         elif self.dataset in [
#             "bloodmnist",
#             "organcmnist",
#             "organsmnist",
#             "dermamnist",
#             "pneumoniamnist",
#         ]:
#             info = medmnist.INFO[self.dataset]
#             DataClass = getattr(medmnist, info["python_class"])
#             dataset = DataClass(root=self.data_root, split=split, transform=transform, download=False)

#             return dataset

#     def get_data(
#         self,
#         train_labeled_indices: list = None,
#         train_unlabeled_indices: list = None,
#         indices_to_fix: list = None,
#         sampling_method: str = None,
#         init_size: int = None,
#         start_iter: bool = True,
#         accelerator: Accelerator = None,
#         labels_per_class: int = None,
#     ):
#         print_fn = accelerator.print if accelerator else print

#         self.train_labeled_indices = train_labeled_indices
#         self.train_unlabeled_indices = train_unlabeled_indices

#         self.full_train = self._dataset_function("train", train=True, augment=False)
#         self.all_train_indices = list(range(len(self.full_train)))

#         """Semi-Supervised Learning"""
#         train_indices = np.array(self.all_train_indices)
#         train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

#         if start_iter:
#             if sampling_method == "equal":
#                 """Equal number of labels per class"""
#                 assert labels_per_class is not None, "labels_per_class must be specified for equal sampling"

#                 self.train_labeled_indices = []
#                 self.train_unlabeled_indices = []

#                 for i in range(self.n_classes):
#                     self.train_labeled_indices.extend(train_indices[train_labels == i][:labels_per_class])
#                     self.train_unlabeled_indices.extend(train_indices[train_labels == i][labels_per_class:])
#             elif sampling_method == "random":
#                 """Random sampling"""
#                 self.train_labeled_indices = np.random.choice(train_indices, init_size, replace=False)
#                 self.train_unlabeled_indices = np.setdiff1d(train_indices, self.train_labeled_indices)
#             else:
#                 """Use all training data"""
#                 self.train_labeled_indices = train_indices
#                 self.train_unlabeled_indices = []
#         else:
#             self.train_labeled_indices = np.append(self.train_labeled_indices, indices_to_fix)
#             indices = np.argwhere(np.isin(self.train_unlabeled_indices, indices_to_fix))
#             self.train_unlabeled_indices = np.delete(self.train_unlabeled_indices, indices)

#         print_fn(f"Current Labeled Train Indices: {len(self.train_labeled_indices)}")
#         print_fn(f"Current Unlabeled Train Indices: {len(self.train_unlabeled_indices)}")

#         self.labeled = Subset(self._dataset_function("train", train=True, augment=True), indices=self.train_labeled_indices)
#         self.unlabeled = Subset(self._dataset_function("train", train=False, augment=False), indices=self.train_unlabeled_indices)
#         self.valid = self._dataset_function("val", train=False, augment=False)

#         self.dload_train = self.create_dataloader(self.full_train, train=True, drop_last=False)
#         self.dload_train_labeled = cycle(self.create_dataloader(self.labeled, train=True))
#         self.dload_train_unlabeled = self.create_dataloader(self.unlabeled, shuffle=False) if len(self.train_unlabeled_indices) > 0 else None
#         self.dload_valid = self.create_dataloader(self.valid, shuffle=False)

#         if accelerator:
#             (
#                 self.dload_train,
#                 self.dload_train_labeled,
#                 self.dload_train_unlabeled,
#                 self.dload_valid,
#             ) = accelerator.prepare(self.dload_train, self.dload_train_labeled, self.dload_train_unlabeled, self.dload_valid)

#         return (
#             self.dload_train,
#             self.dload_train_labeled,
#             self.dload_train_unlabeled,
#             self.dload_valid,
#             self.train_labeled_indices,
#             self.train_unlabeled_indices,
#         )

#     def create_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = False, train: bool = False):
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size if train else 250,
#             shuffle=shuffle,
#             num_workers=0,
#             drop_last=drop_last,
#             pin_memory=True,
#         )

#     def get_test_data(self):
#         self.test = self._dataset_function("test", train=False, augment=False)
#         self.dload_test = self.create_dataloader(self.test, shuffle=False, drop_last=False)

#         return self.dload_test

#     def query_samples(
#         self,
#         f: nn.Module,
#         dload_train_unlabeled: DataLoader,
#         train_unlabeled_inds: list[int],
#         query_size: int,
#         accelerator: Accelerator = None,
#     ):
#         confs, confs_to_fix = [], []

#         f.eval()
#         progress_bar = tqdm(
#             dload_train_unlabeled,
#             desc="Predicting",
#             disable=not accelerator.is_main_process if accelerator else True,
#         )
#         device = accelerator.device if accelerator else t.device("cuda")
#         print_fn = accelerator.print if accelerator else print

#         for _, (x_p_d, y_p_d) in enumerate(progress_bar):
#             x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device).squeeze().long()

#             with t.no_grad():
#                 logits = accelerator.unwrap_model(f).classify(x_p_d) if accelerator else f.classify(x_p_d)

#             confs.extend(nn.functional.softmax(logits, dim=1).detach().cpu().numpy())

#         confs = np.array(confs).reshape((-1, self.n_classes))

#         for ind, conf in enumerate(confs):
#             if ind >= len(train_unlabeled_inds):
#                 break
#             confs_to_fix.append((conf.max(), train_unlabeled_inds[ind]))

#         query_size = min(query_size, len(train_unlabeled_inds))

#         confs_to_fix.sort(key=lambda x: x[0])
#         confs_to_fix = confs_to_fix[:query_size]
#         inds_to_fix = [ind for _, ind in confs_to_fix]
#         inds_to_fix.sort()

#         print_fn(f"Length of inds to fix: {len(inds_to_fix)}")

#         return inds_to_fix

#     def get_class_distribution(self):
#         class_labels = [self.classnames[self.full_train[idx][1][0]] for idx in self.train_labeled_indices]
#         num_samples_added_per_class = defaultdict(int)

#         for label in class_labels:
#             num_samples_added_per_class[label] += 1

#         counts = sorted(num_samples_added_per_class.items(), key=lambda x: x[0])

#         return counts

#     def get_full_distribution(self):
#         class_labels = [self.classnames[self.full_train[idx][1][0]] for idx in self.all_train_indices]
#         num_samples_added_per_class = defaultdict(int)

#         for label in class_labels:
#             num_samples_added_per_class[label] += 1

#         counts = sorted(num_samples_added_per_class.items(), key=lambda x: x[0])

#         return counts

import os

import medmnist
import numpy as np
import torch as t
import torch.functional as F
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SUPPORTED_DATASETS = ["bloodmnist", "organcmnist", "organsmnist", "dermamnist", "pneumoniamnist"]


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
        if self.dataset in SUPPORTED_DATASETS:
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

        if self.dataset in SUPPORTED_DATASETS:
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
    ):
        self.full_train = self._dataset_function("train", train=True, augment=False)
        self.all_train_indices = list(range(len(self.full_train)))

        # Semi-Supervised Learning (SSL)
        train_indices = np.array(self.all_train_indices)
        train_labels = np.array([np.squeeze(self.full_train[ind][1]) for ind in train_indices])

        if start_iter:
            # Equal number of labels per class
            if sample_method == "equal":
                assert labels_per_class is not None, "labels_per_class must be specified for equal sampling."

                self.labeled_indices = []
                self.unlabeled_indices = []

                for i in range(self.n_classes):
                    self.labeled_indices.extend(train_indices[train_labels == i][:labels_per_class])
                    self.unlabeled_indices.extend(train_indices[train_labels == i][labels_per_class:])

            # Random sampling
            elif sample_method == "random":
                assert init_size is not None, "init_size must be specified for random sampling."

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
        self.unlabeled = DataSubset(self._dataset_function("train", train=False, augment=False), indices=self.unlabeled_indices)
        self.val = self._dataset_function("val", train=False, augment=False)

    def query(self, f: t.nn.Module, unlabeled_dataloader: DataLoader, query_size: int):
        progress_bar = tqdm(unlabeled_dataloader, desc="Prediction Progress")
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        f.to(device)

        f.eval()
        confs = []
        for x, y in enumerate(progress_bar):
            x, y = x.to(device), y.to(device).squeeze().long()

            with t.no_grad():
                logits = f.classify(x)

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            confidences = t.max(probs, dim=1)[0]

            confs.append(confidences.cpu().numpy())

        confs = np.array(confs)
        sorted_indices = np.argsort(confs)
        selected_indices = sorted_indices[:query_size]

        self.labeled_indices = np.append(self.labeled_indices, selected_indices)
        indices = np.argwhere(np.isin(self.unlabeled_indices, selected_indices))
        self.unlabeled_indices = np.delete(self.unlabeled_indices, indices)

        self.labeled = DataSubset(self._dataset_function("train", train=True, augment=True), indices=self.labeled_indices)
        self.unlabeled = DataSubset(self._dataset_function("train", train=False, augment=False), indices=self.unlabeled_indices)

        print("Current labeled train indices:", len(self.labeled_indices))
        print("Current unlabeled train indices:", len(self.unlabeled_indices))

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
