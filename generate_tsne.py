import torch as t
from utils import DataModule, load_config
from accelerate import Accelerator
from pathlib import Path
import argparse
from models.JEM import get_model_and_buffer
from sklearn.manifold import TSNE
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_tsne(model, dload_train, device, n_samples=1000, n_components=2, perplexity=30, n_iter=1000, lr=200, random_state=0):
    """
    Returns the t-SNE embedding of the model's latent space.
    """

    # generate features
    features = []
    labels = []

    model = model.to("cuda")
    for i, batch in enumerate(tqdm(dload_train, desc="Generating features")):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        features.append(model.feature(x).detach().cpu())
        labels.append(y.detach().cpu())

    features = t.cat(features, dim=0)
    labels = t.cat(labels, dim=0)

    # sample
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(features)

    df = pd.DataFrame(tsne_output, columns=["x", "y"])
    df["targets"] = labels
    df["targets"] = df["targets"].apply(lambda x: datamodule.classnames[x])

    plt.rcParams["figure.figsize"] = 10, 10
    scatter_plot = sns.scatterplot(
        x="x",
        y="y",
        hue="targets",
        palette=sns.color_palette("hls", datamodule.n_classes),
        data=df,
        marker="o",
        legend="full",
        alpha=0.75,
    )
    scatter_plot.legend(bbox_to_anchor=(1.05, 1), loc="upper right")

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")

    plt.savefig(os.path.join("./", f"tsne_{datamodule.dataset}.png"), bbox_inches="tight")
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Active Learning with JEM++")

    parser.add_argument("--model_config", type=str, default="configs/jempp_hparams.yml", help="Path to the config file.")
    parser.add_argument("--dataset_config", type=str, default="configs/cifar10.yml", help="Path to the config file.")
    args = parser.parse_args()

    model_config = load_config(Path(args.model_config))
    dataset_config = load_config(Path(args.dataset_config))
    config = {**model_config, **dataset_config}

    accelerator = Accelerator()
    datamodule = DataModule(accelerator=accelerator, **config)
    dload_train, dload_train_labeled, dload_train_unlabeled, dload_valid, train_labeled_inds, train_unlabeled_inds = datamodule.get_data()

    f, replay_buffer = get_model_and_buffer(accelerator=accelerator, datamodule=datamodule, **config)

    ckpt = t.load("/home/lorenzo/repos/JEMPP/runs/2023-12-06_21-11-29_bloodmnist_l760ez83/checkpoints/baseline_6400/epoch=77-val_loss=0.1389.ckpt")
    f.load_state_dict(ckpt["model_state_dict"])

    # Get the t-SNE embedding of the latent space
    z_tsne = get_tsne(
        f,
        dload_train,
        accelerator.device,
        n_samples=1000,
        n_components=2,
        perplexity=30,
        n_iter=1000,
        lr=200,
        random_state=0,
    )
