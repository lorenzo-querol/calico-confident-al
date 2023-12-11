import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_config


def get_test_metrics(experiment_name):
    experiment_path = os.path.join(os.getcwd(), "runs", experiment_name, "test")
    folders = os.listdir(experiment_path)

    test_metrics = []
    for folder in folders:
        file_path = os.path.join(experiment_path, folder, "test_metrics.csv")
        df = pd.read_csv(file_path)
        df["iter"] = folder
        test_metrics.append(df)

    test_metrics = pd.concat(test_metrics)

    al = test_metrics[test_metrics["iter"].str.contains("al")]
    baseline = test_metrics[test_metrics["iter"].str.contains("baseline")]

    for df in [al, baseline]:
        df["iter"] = df["iter"].str.extract("(\d+)").astype(int)
        df.sort_values(by="iter", inplace=True)

    min_length = min(len(al), len(baseline))
    al = al.iloc[:min_length]
    baseline = baseline.iloc[:min_length]

    al["iter"] = baseline["iter"].tolist()

    al.set_index(np.arange(len(al)), inplace=True)
    baseline.set_index(np.arange(len(baseline)), inplace=True)

    return al, baseline


def get_metrics(df):
    test_loss, test_acc, ece = df["test_loss"].tolist(), df["test_acc"].tolist(), df["test_ece"].tolist()
    return test_loss, test_acc, ece


def plot_acc_ece(df, df2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting accuracy against iteration for df on the first subplot
    df.plot(x="iter", y="test_acc", marker="o", label="Test Accuracy (df)", ax=ax1)
    ax1.set_title("Test Accuracy vs Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Test Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Plotting accuracy against iteration for df2 on the first subplot
    df2.plot(x="iter", y="test_acc", marker="x", label="Test Accuracy (df2)", ax=ax1)
    ax1.legend()  # Keep legends updated
    ax1.grid(True)

    # Plotting ECE against iteration for df on the second subplot
    df.plot(x="iter", y="test_ece", marker="o", color="orange", label="ECE (df)", ax=ax2)
    ax2.set_title("ECE vs Iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("ECE")
    ax2.legend()
    ax2.grid(True)

    # Plotting ECE against iteration for df2 on the second subplot
    df2.plot(x="iter", y="test_ece", marker="x", color="red", label="ECE (df2)", ax=ax2)
    ax2.legend()  # Keep legends updated
    ax2.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig("test.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Active Learning with JEM++")

    parser.add_argument("--dataset_config", type=str, default="configs/cifar10.yml", help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(Path(args.dataset_config))

    if config["experiment_name"] is None:
        assert False, "Please specify experiment name"

    df_al, df_baseline = get_test_metrics(config["experiment_name"])
    print(df_al)
    print(df_baseline)
    # plot_acc_ece(df_al, df_baseline)
