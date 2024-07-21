from utils import create_log_dir, initialize, parse_args, write_to_yaml
from models.JEM import get_model
from tensorboardX import SummaryWriter
from DataModule import MedMNISTDataModule
from train_helpers import fit


# def test(f, datamodule: MedMNISTDataModule, path: str, num_labeled: int):
#     device = t.device("cuda" if t.cuda.is_available() else "cpu")
#     f.to(device)

#     test_dataloader = datamodule.test_dataloader()

#     progress_bar = tqdm(test_dataloader, desc="Test Progress", total=len(test_dataloader), position=0, leave=True)

#     f.eval()
#     test_loss, test_acc, all_confs, all_gts = [], [], [], []
#     for x_lab, y_lab in progress_bar:
#         x_lab, y_lab = x_lab.to(device), y_lab.to(device).squeeze().long()

#         with t.no_grad():
#             logits = f.classify(x_lab)
#             ce_loss = nn.functional.cross_entropy(logits, y_lab, reduction="none").detach().cpu().numpy()
#             confs = nn.functional.softmax(logits, dim=1).float().cpu().numpy()

#         acc = (logits.max(1)[1] == y_lab).float().cpu().numpy()
#         gts = y_lab.detach().cpu().numpy()

#         test_loss.extend(ce_loss)
#         test_acc.extend(acc)
#         all_confs.extend(confs)
#         all_gts.append(gts)

# Average loss and accuracy over the test set
# test_loss = np.mean(test_loss)
# test_acc = np.mean(test_acc)
# test_ece = compute_ece(
#     all_confs,
#     all_gts,
#     datamodule.n_classes,
#     save_rel_diagram=True,
#     save_path=f"{path}/num-labels-{num_labeled}-reliability.tikz",
# )

# # Save as a CSV file
# values = {
#     "num_labeled": [num_labeled],
#     "test_loss": [test_loss.item()],
#     "test_acc": [test_acc.item()],
#     "test_ece": [test_ece],
# }
# test_df = pd.DataFrame(values)

# if os.path.exists(f"{path}/test_metrics.csv"):
#     test_df.to_csv(f"{path}/test_metrics.csv", mode="a", header=False, index=False)
# else:
#     test_df.to_csv(f"{path}/test_metrics.csv", mode="w", header=True, index=False)


if __name__ == "__main__":
    args = parse_args()
    initialize(args.seed)

    log_dir = f"./{args.log_dir}/{args.dataset}/{args.exp_name}"
    log_dir = create_log_dir(log_dir)
    writer = SummaryWriter(log_dir)

    write_to_yaml(vars(args), f"{log_dir}/config.yaml")

    datamodule = MedMNISTDataModule(args.dataset, args.root_dir, args.batch_size, args.sigma, args.seed)
    datamodule.setup(sample_method="random", init_size=args.query_size, log_dir=log_dir)

    LIMIT = 4000
    ITERATIONS = LIMIT // args.query_size

    for i in range(ITERATIONS):
        print(f"\nIteration {i+1}")

        f = get_model(datamodule, args)
        fit(f, datamodule, args, writer, log_dir, i)

        if len(datamodule.labeled_indices) != LIMIT:
            indices_to_fix = datamodule.query(f, args.query_size)
            datamodule.setup(indices_to_fix=indices_to_fix, log_dir=log_dir, start_iter=False)

    writer.close()
