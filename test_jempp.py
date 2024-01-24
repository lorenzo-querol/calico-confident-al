import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from accelerate.utils import set_seed
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import F
from utils import get_directories, parse_args


def test_model(f: nn.Module, datamodule: DataModule, test_dir: str, num_labeled: int):
    dload_test = datamodule.get_test_data()

    all_corrects, all_losses = [], []
    all_confs, all_gts = [], []
    test_loss, test_acc = np.inf, 0.0

    correct_per_class = {label: 0 for label in datamodule.classnames}
    total_per_class = {label: 0 for label in datamodule.classnames}

    f.eval()
    progress_bar = tqdm(dload_test, desc="Testing")
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to("cuda"), labels.to("cuda").squeeze().long()

        with t.no_grad():
            logits = f.classify(inputs)

        loss, correct, confs, targets = (
            t.nn.functional.cross_entropy(logits, labels, reduction="none"),
            (logits.max(1)[1] == labels).float(),
            t.nn.functional.softmax(logits, dim=1),
            labels,
        )

        all_gts.extend(targets)
        all_confs.extend(confs)
        all_losses.extend(loss)
        all_corrects.extend(correct)

        for i, class_name in enumerate(datamodule.classnames):
            correct_per_class[class_name] += t.sum((correct == 1) & (targets == i)).item()
            total_per_class[class_name] += t.sum(targets == i).item()

    test_loss = np.mean([loss.item() for loss in all_losses])
    test_acc = np.mean([correct.item() for correct in all_corrects])

    accuracy_per_class = {
        label: correct / total if total > 0 else 0
        for label, (correct, total) in zip(datamodule.classnames, zip(correct_per_class.values(), total_per_class.values()))
    }

    all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape((-1, datamodule.n_classes))
    all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

    bins = 10
    ece, diagram = ECE(bins), ReliabilityDiagram(bins)
    calibration_score = ece.measure(all_confs, all_gts)
    pl = diagram.plot(all_confs, all_gts, dpi=600)

    test_metrics = {"test_loss": test_loss, "test_acc": test_acc, "test_ece": calibration_score}
    test_metrics = pd.DataFrame(test_metrics, index=[0])

    accuracy_per_class = pd.DataFrame(accuracy_per_class, index=[0])

    class_distribution = datamodule.get_class_distribution()
    class_distribution = pd.DataFrame([dict(class_distribution)], index=[0])

    full_distribution = datamodule.get_full_distribution()
    full_distribution = pd.DataFrame([dict(full_distribution)], index=[0])

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    pl.savefig(f"{test_dir}/reliability_diagram.png")
    plt.close(pl)

    test_metrics.to_csv(f"{test_dir}/test_metrics.csv", index=False)
    accuracy_per_class.to_csv(f"{test_dir}/accuracy_per_class.csv", index=False)
    class_distribution.to_csv(f"{test_dir}/class_distribution.csv", index=False)
    full_distribution.to_csv(f"{test_dir}/full_distribution.csv", index=False)

    print(f"Test Loss: {test_loss} | Test Accuracy: {test_acc} | ECE: {calibration_score}")


def get_ckpts(ckpt_dir: str, experiment_type: str):
    ckpts = list(Path(ckpt_dir).rglob("*"))
    ckpts = [ckpt for ckpt in ckpts if not ckpt.is_dir()]
    ckpts = [ckpt for ckpt in ckpts if "last" in ckpt.name]
    ckpts = [str(ckpt) for ckpt in ckpts]
    ckpts = [ckpt for ckpt in ckpts if experiment_type in ckpt]

    ckpt_dicts = []
    for ckpt in ckpts:
        path = ckpt.split("/")
        experiment_type, calibrated, optim, num_labeled = path[-2].split("_")
        experiment_type = f"{experiment_type}_{calibrated}_{optim}"
        ckpt_dicts.append({"experiment_type": experiment_type, "path": ckpt, "num_labeled": int(num_labeled)})

    ckpt_dicts = sorted(ckpt_dicts, key=lambda x: int(x["num_labeled"]))

    return ckpt_dicts


limit_dict = {
    "cifar10": 40000,
    "cifar100": 40000,
    "svhn": 40000,
    "bloodmnist": 4000,
    "dermamnist": 4000,
    "pneumoniamnist": 4000,
    "organsmnist": 4000,
    "organcmnist": 4000,
}

equal_dict = {
    "pneumoniamnist": 2400,
    "bloodmnist": 6400,
}


def main(config):
    datamodule = DataModule(accelerator=None, **config)
    datamodule.prepare_data()

    (
        dload_train,
        dload_train_labeled,
        dload_train_unlabeled,
        dload_valid,
        train_labeled_inds,
        train_unlabeled_inds,
    ) = datamodule.get_data(
        sampling_method="random" if config["labels_per_class"] <= 0 else "equal",
        init_size=config["query_size"],
        labels_per_class=config["labels_per_class"],
    )
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    ckpt_dir, _, test_dir = get_directories(**config)
    limit = limit_dict[config["dataset"]] if config["labels_per_class"] <= 0 else equal_dict[config["dataset"]]
    labels_per_class = config["labels_per_class"]

    f = F(n_channels=datamodule.img_shape[0], n_classes=datamodule.n_classes, **config).to(device)

    for ckpt_dict in get_ckpts(ckpt_dir, config["experiment_type"]):
        experiment_type, path, num_labeled = ckpt_dict.values()

        """Load the best checkpoint"""
        print(f"Loading best checkpoint from {path}.")
        f.load_state_dict(t.load(path)["model_state_dict"])
        f.to(device)

        """---TESTING---"""
        folder_name = f"{test_dir}/{experiment_type}_{num_labeled}"
        test_model(f=f, datamodule=datamodule, test_dir=folder_name, num_labeled=num_labeled)

        if len(train_labeled_inds) >= limit:
            break

        # Least confident sampling
        if config["labels_per_class"] == 0:
            print(f"Querying {config['query_size']} samples using least confident sampling.")
            inds_to_fix = datamodule.query_samples(
                f,
                dload_train_unlabeled,
                train_unlabeled_inds,
                config["query_size"],
            )
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(
                train_labeled_inds,
                train_unlabeled_inds,
                inds_to_fix,
                start_iter=False,
            )

        # Equal labels sampling
        elif config["labels_per_class"] > 0:
            print(f"Querying {config['labels_per_class']} samples per class.")
            labels_per_class += config["labels_per_class"]
            (
                dload_train,
                dload_train_labeled,
                dload_train_unlabeled,
                dload_valid,
                train_labeled_inds,
                train_unlabeled_inds,
            ) = datamodule.get_data(
                train_labeled_inds,
                train_unlabeled_inds,
                sampling_method="equal",
                labels_per_class=labels_per_class,
            )


if __name__ == "__main__":
    config = vars(parse_args())
    set_seed(config["seed"])

    """Scale batch size by number of GPUs for reproducibility"""
    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})
    config.update({"p_x_weight": 1.0 if config["calibrated"] else 0.0})

    main(config)
