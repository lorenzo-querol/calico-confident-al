import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from accelerate.utils import set_seed
from sklearn.manifold import TSNE
from tqdm import tqdm

from DataModule import DataModule
from models.JEM import get_model, get_optim
from utils import parse_args

args = parse_args()
set_seed(args.seed)


datamodule = DataModule(dataset=args.dataset, root_dir=args.root_dir, batch_size=args.batch_size, sigma=args.sigma)
datamodule.test_setup(test_dir=args.log_dir)
dload_test = datamodule.test_dataloader()

# ckpt_path = None
ckpt_path = "runs/pneumoniamnist/baseline-softmax/checkpoints/num-labels-4000_last.ckpt"
f = get_model(datamodule, args, ckpt_path)
device = "cuda" if t.cuda.is_available() else "cpu"
f = f.to(device)

features = []
labels = []

for i, (x, y) in enumerate(tqdm(dload_test, desc="Generating features")):
    x, y = x.to(device), y.to(device)
    features.append(f.feature(x).detach().cpu())
    labels.append(y.detach().cpu())

features = t.cat(features, dim=0)
labels = t.cat(labels, dim=0)

tsne = TSNE(random_state=args.seed)
tsne_output = tsne.fit_transform(features)


df = pd.DataFrame(tsne_output, columns=["x", "y"])
df["targets"] = labels
# df["targets"] = df["targets"].apply(lambda x: datamodule.classes[x])

# Define a colormap
cmap = plt.cm.get_cmap("viridis", len(df["targets"].unique()))

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

scatter_plot = ax.scatter(x=df["x"], y=df["y"], s=1, c=df["targets"].astype("category").cat.codes, cmap=cmap, alpha=0.8)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f'{args.dataset} {ckpt_path.split("/")[2]}', fontsize=14)

targets = df["targets"].unique()
colors = [cmap(i) for i in range(len(targets))]

patches = [matplotlib.patches.Patch(color=colors[i], label=target) for i, target in enumerate(targets)]

ax.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), loc="upper center", ncol=5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(f'plots/{args.dataset}_{ckpt_path.split("/")[2]}_tsne.png')
