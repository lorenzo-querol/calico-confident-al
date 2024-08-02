import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import F
from utils import initialize, parse_args 
from torch.utils.data import DataLoader
 
def save_rel_diagram(pl, test_dir):
    if isinstance(pl, str):
        with open(f"{test_dir}/reliability_diagram.tikz", "w") as f:
            print(f'Writing to file "{test_dir}/reliability_diagram.tikz".')
            f.write(pl)
    else:
        pl.savefig(f"{test_dir}/reliability_diagram.png")
        plt.close(pl)

def get_ckpt_dicts(ckpt_dir:str):
    ckpts = list(Path(ckpt_dir).rglob("*"))
    ckpts = [c for c in ckpts if "last" in c.name]

    ckpt_dicts = []
    for c in ckpts:
        name = os.path.splitext(os.path.basename(c))[0]
        al_iter, _, _ = name.split("-")
        ckpt_dicts.append({"path": c, "al_iter": int(al_iter)})

    return ckpt_dicts

def test(f: nn.Module, dload_test:DataLoader, n_classes:int, device,test_dir: str):
    all_corrects, all_losses = [], []
    all_confs, all_gts = [], []
    test_loss, test_acc, test_ece = np.inf, 0.0, np.inf
    ece, diagram = ECE(10), ReliabilityDiagram(10)
 

    f.eval()
    for i, (x, y) in enumerate(tqdm(dload_test, desc="Testing")):
        x, y = x.to(device), y.to(device).squeeze().long()

        with t.no_grad():
            logits = f.classify(x)

        loss, correct, confs, targets = (
            t.nn.functional.cross_entropy(logits, y, reduction="none"),
            (logits.max(1)[1] == y).float(),
            t.nn.functional.softmax(logits, dim=1),
            y,
        )

        all_gts.extend(targets)
        all_confs.extend(confs) 
        all_losses.extend(loss)
        all_corrects.extend(correct)


    test_loss = np.mean([loss.item() for loss in all_losses])
    test_acc = np.mean([correct.item() for correct in all_corrects])


    all_confs = np.array([conf.cpu().numpy() for conf in all_confs]).reshape((-1, n_classes))
    all_gts = np.array([gt.cpu().numpy() for gt in all_gts])

    test_ece = ece.measure(all_confs, all_gts)
    pl = diagram.plot(all_confs, all_gts, tikz=True)

    test_metrics = {"test_loss": test_loss, "test_acc": test_acc, "test_ece": test_ece}
    test_metrics = pd.DataFrame(test_metrics, index=[0])


    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    save_rel_diagram(pl, test_dir)

    test_metrics.to_csv(f"{test_dir}/test_metrics.csv", index=False)

    print(f"Test Loss: {test_loss} | Test Accuracy: {test_acc} | ECE: {test_ece}")



    
def main(config):
    initialize(config["seed"])
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    datamodule = DataModule(**config)
    datamodule.prepare_data()
    dload_test = datamodule.get_test_data()

    ckpt_dir = f'{config['log_dir']}/checkpoints'
    test_dir = f'{config['log_dir']}/test'
    
    f = F(n_channels=datamodule.img_shape[0], n_classes=datamodule.n_classes, **config)
    f = f.to(device)
    
    ckpt_dicts = get_ckpt_dicts(ckpt_dir)
    for cd in ckpt_dicts:
        f.load_state_dict(t.load(cd["path"])["model_state_dict"])
        test(f, dload_test, datamodule.n_classes, device, f'{test_dir}/{cd["al_iter"]}' )
 
if __name__ == "__main__":
    config = vars(parse_args())
    main(config)
