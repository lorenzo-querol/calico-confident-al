import torch
from torch.utils.data import random_split
from torchvision import datasets


class OtherDataset:
    def __init__(self, dataset_name, root="./data", split=None, transform=None, download=True):
        """
        Initialize the SplitCIFARDataset.

        Parameters:
        - dataset_name (str): Name of the dataset ('cifar10', 'cifar100', or 'svhn').
        - root (str): Root directory to store the datasets.
        - transform (torchvision.transforms.Compose): Transformation applied to the training set.
        - download (bool): Whether to download the dataset.
        """

        self.dataset_name = dataset_name
        self.root = root
        self.split = split
        self.transform = transform
        self.download = download

        # Load the specified dataset
        if dataset_name == "mnist":
            self.dataset = datasets.MNIST(root=self.root, train=True if split == "train" else False, download=download, transform=transform)
            self.classes = self.dataset.classes
        if dataset_name == "cifar10":
            self.dataset = datasets.CIFAR10(root=self.root, train=True if split == "train" else False, download=download, transform=transform)
            self.classes = self.dataset.classes
        elif dataset_name == "cifar100":
            self.dataset = datasets.CIFAR100(root=self.root, train=True if split == "train" else False, download=download, transform=transform)
            self.classes = self.dataset.classes
        elif dataset_name == "svhn":
            self.dataset = datasets.SVHN(root=self.root, split=split, download=download, transform=transform)
            self.classes = self.dataset.classes
        else:
            raise ValueError("Invalid dataset_name. Choose 'cifar10', 'cifar100', or 'svhn'.")

        self.train_dataset, self.val_dataset = self._split_dataset()
        self.get_dataset()

    def _split_dataset(self, validation_ratio=0.2, shuffle=True, random_seed=42):
        """
        Split the dataset into training and validation sets.

        Parameters:
        - validation_ratio (float): Ratio of the dataset to be used for validation.
        - shuffle (bool): Whether to shuffle the dataset before splitting.
        - random_seed (int): Seed for reproducibility.

        Returns:
        - train_dataset (torch.utils.data.Dataset): Training dataset.
        - val_dataset (torch.utils.data.Dataset): Validation dataset.
        """
        dataset_size = len(self.dataset)
        val_size = int(validation_ratio * dataset_size)
        train_size = dataset_size - val_size

        # Set random seed for reproducibility
        torch.manual_seed(random_seed)

        # Perform the random split
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

        return train_dataset, val_dataset

    def get_dataset(self):
        if self.split == "train":
            return self.train_dataset
        elif self.split == "val":
            return self.val_dataset
        elif self.split == "test":
            return self.dataset
