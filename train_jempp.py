import os

import numpy as np
import torch as t
import torch.nn as nn
from accelerate.utils import set_seed
from modAL import ActiveLearner
from skorch import NeuralNetClassifier

from CustomTrainer import CustomTrainer
from DataModule import DataModule
from models.JEM import F
from sam import SAM
from utils import get_args


def category_mean(datamodule: DataModule):
    dataset = datamodule.dataset
    img_shape = datamodule.img_shape
    n_classes = datamodule.n_classes
    train_dataloader = datamodule.full_train

    centers = t.zeros([n_classes, int(np.prod(img_shape))])
    covs = t.zeros([n_classes, int(np.prod(img_shape)), int(np.prod(img_shape))])

    im_test, targ_test = [], []
    for im, targ in train_dataloader:
        im_test.append(im)
        targ_test.append(targ)

    im_test, targ_test = t.cat(im_test), t.cat(targ_test)

    for i in range(n_classes):
        if datamodule.dataset in ["cifar10", "cifar100", "svhn", "mnist"]:
            mask = targ_test == i
        else:
            mask = (targ_test == i).squeeze(1)

        imc = im_test[mask]
        imc = imc.view(len(imc), -1)
        mean = imc.mean(dim=0)
        sub = imc - mean.unsqueeze(dim=0)
        cov = sub.t() @ sub / len(imc)
        centers[i] = mean
        covs[i] = cov

    if not os.path.exists("weights"):
        os.makedirs("./weights")

    t.save(centers, f"weights/{dataset}_mean.pt")
    t.save(covs, f"weights/{dataset}_cov.pt")


def get_optim(model: nn.Module, args):
    optimizers = {"adam": t.optim.Adam, "sgd": t.optim.SGD}

    if args.optim not in optimizers:
        raise ValueError(f"Optimizer {args.optim} not supported.")

    base_optim = optimizers[args.optim]

    if args.sam:
        return SAM(
            model.parameters(),
            base_optimizer=base_optim,
            rho=args.rho,
            adaptive=True if args.rho > 0.5 else False,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optim_params = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}

        if args.optim == "sgd":
            optim_params["momentum"] = 0.9
            optim_params["nesterov"] = True

        return base_optim(**optim_params)


def get_model(datamodule: DataModule, args):
    return F(args.depth, args.width, datamodule.n_classes, datamodule.img_shape[0], args.norm)


def train_model(args):
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using {device} device {t.cuda.get_device_name(0)}.")
    datamodule = DataModule(dataset=args.dataset, root_dir=args.root_dir, batch_size=args.batch_size, sigma=args.sigma)

    if not os.path.isfile(f"weights/{datamodule.dataset}_cov.pt"):
        category_mean(datamodule=datamodule)

    f = get_model(datamodule, args)
    optimizer = get_optim(f, args)

    trainer = CustomTrainer(f, optimizer, datamodule, device, args)

    if args.test:
        trainer.test(ckpt_path=args.ckpt_path)
        return

    trainer.fit()


if __name__ == "__main__":
    t.backends.cudnn.enabled = True
    t.backends.cudnn.benchmark = True
    t.backends.cudnn.deterministic = False

    args = get_args()
    set_seed(args.seed)
    train_model(args)
