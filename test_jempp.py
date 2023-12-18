import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import get_model_and_buffer
from utils import get_directories, get_experiment_name, get_logger_kwargs, parse_args


def test_model(f: nn.Module, accelerator: Accelerator, datamodule: DataModule, test_dir: str, num_labeled: int = None, **config):
    dload_test = datamodule.get_test_data()

    all_corrects, all_losses = [], []
    all_confs, all_gts = [], []
    test_loss, test_acc = np.inf, 0.0

    correct_per_class = {label: 0 for label in datamodule.classnames}
    total_per_class = {label: 0 for label in datamodule.classnames}

    progress_bar = tqdm(dload_test, desc="Testing", disable=not accelerator.is_main_process)
    for i, (inputs, labels) in enumerate(progress_bar):
        labels = labels.squeeze().long()

        with t.no_grad():
            logits = accelerator.unwrap_model(f).classify(inputs)

        loss, correct, confs, targets = accelerator.gather_for_metrics(
            (
                t.nn.functional.cross_entropy(logits, labels),
                (logits.max(1)[1] == labels).float(),
                t.nn.functional.softmax(logits, dim=1),
                labels,
            )
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
        for label, (correct, total) in zip(
            datamodule.classnames,
            zip(correct_per_class.values(), total_per_class.values()),
        )
    }

    all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape((-1, datamodule.n_classes))
    all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

    ece, diagram = ECE(10), ReliabilityDiagram(10)
    calibration_score = ece.measure(all_confs, all_gts)
    pl = diagram.plot(all_confs, all_gts)

    test_metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_ece": calibration_score,
    }
    test_metrics = pd.DataFrame(test_metrics, index=[0])

    accuracy_per_class = pd.DataFrame(accuracy_per_class, index=[0])

    class_distribution = datamodule.get_class_distribution()
    class_distribution = pd.DataFrame(class_distribution, columns=["Class", "Num Samples"])

    if accelerator.is_main_process:
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)

        pl.savefig(f"{test_dir}/reliability_diagram.png")
        plt.close()
        test_metrics.to_csv(f"{test_dir}/test_metrics.csv", index=False)
        accuracy_per_class.to_csv(f"{test_dir}/accuracy_per_class.csv", index=False)
        class_distribution.to_csv(f"{test_dir}/class_distribution.csv", index=False)

    accelerator.print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | ECE: {calibration_score:.4f}")

    if config["enable_tracking"]:
        accelerator.log(
            {
                "num_labeled": num_labeled,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_ece": calibration_score,
            }
        )


def get_ckpts(ckpt_dir):
    ckpts = list(Path(ckpt_dir).rglob("*"))
    ckpts = [ckpt for ckpt in ckpts if not ckpt.is_dir()]
    ckpts = [ckpt for ckpt in ckpts if "last" not in ckpt.name]
    ckpts = [str(ckpt) for ckpt in ckpts]

    ckpt_dicts = []
    for ckpt in ckpts:
        path = ckpt.split("/")
        experiment_type, num_labeled = path[-2].split("_")

        ckpt_dicts.append({"experiment_type": experiment_type, "path": ckpt, "num_labeled": num_labeled})

    ckpt_dicts = sorted(ckpt_dicts, key=lambda x: int(x["num_labeled"]))

    return ckpt_dicts


def main(config):
    accelerator = Accelerator(log_with="wandb" if config["enable_tracking"] else None)
    datamodule = DataModule(accelerator=accelerator, **config)
    datamodule.prepare_data()

    config.update("experiment_name", get_experiment_name(**config))
    ckpt_dir, _, test_dir = get_directories(**config)

    for ckpt_dict in get_ckpts(ckpt_dir):
        if config["enable_tracking"]:
            logger_kwargs = get_logger_kwargs(**config)
            init_kwargs = {
                "wandb": {
                    "tags": [f'seed_{config["seed"]}', "test"],
                    **logger_kwargs,
                }
            }
            accelerator.init_trackers(project_name="JEM", config=config, init_kwargs=init_kwargs)

        """Load the best checkpoint"""
        accelerator.print(f"Loading best checkpoint from {ckpt_dict['path']}.")
        f, _ = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, load_path=ckpt_dict["path"], **config)

        """---TESTING---"""
        test_model(
            f=f,
            accelerator=accelerator,
            datamodule=datamodule,
            test_dir=f"{test_dir}/{ckpt_dict['experiment_type']}_{ckpt_dict['num_labeled']}",
            num_labeled=ckpt_dict["num_labeled"],
            **config,
        )

    if config["enable_tracking"]:
        accelerator.end_training()


if __name__ == "__main__":
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.enabled = True
    t.backends.cudnn.deterministic = True

    config = vars(parse_args())

    """Scale batch size by number of GPUs for reproducibility"""
    config.update({"batch_size": config["batch_size"] // t.cuda.device_count()})
    config.update("p_x_weight", 1.0 if config["calibrated"] else 0.0)

    set_seed(config["seed"])

    main(config)
