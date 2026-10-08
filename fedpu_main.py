"""Standalone runner for the ICML 2022 FedPU binary specialization."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from datasets.dataset import load_dataset
from other.baseline_runtime import BaselineCNN, device, evaluate_two_logit, parse_train_data
from other.fedpu_pu import local_train_fedpu
from other.nnpu_common import average_models, clone_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone binary FedPU")
    parser.add_argument("--dataset", default="cifar10",
                        choices=("cifar10", "fashion_mnist", "chest", "kolektor_sdd2"))
    parser.add_argument("--chest_root", default=None)
    parser.add_argument("--chest_val_root", default=None)
    parser.add_argument("--kolektor_root", default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--local_epochs", type=int, default=6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pu_weight", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    if min(args.rounds, args.n_clients, args.batch_size,
           args.local_epochs, args.max_steps_per_epoch) < 1:
        raise ValueError("positive arguments required")
    if args.batch_size < 2 or args.alpha <= 0 or args.lr <= 0 or args.pu_weight < 0:
        raise ValueError("invalid FedPU settings")

    sizes = {
        "cifar10": (1000, 47976, 1024, 3),
        "fashion_mnist": (1000, 57976, 1024, 1),
        "chest": (1000, 3960, 256, 1),
        "kolektor_sdd2": (120, 1900, 256, 3),
    }
    n_labeled, n_unlabeled, n_proxy, channels = sizes[args.dataset]
    kwargs = {"alpha": args.alpha, "n_proxy": n_proxy}
    if args.dataset == "chest":
        if not args.chest_root or not args.chest_val_root:
            raise ValueError("chest roots are required")
        kwargs.update(chest_root=args.chest_root,
                      chest_val_root=args.chest_val_root)
    elif args.dataset == "kolektor_sdd2":
        if not args.kolektor_root:
            raise ValueError("--kolektor_root is required")
        kwargs["kolektor_root"] = args.kolektor_root

    raw, test_raw, prior_values, proxy_raw = load_dataset(
        args.dataset, n_labeled, n_unlabeled, args.n_clients, **kwargs
    )
    del proxy_raw
    clients = [parse_train_data(records) for records in raw]
    x_test, t_test = parse_train_data(test_raw)
    eps = np.finfo(np.float64).eps
    priors = np.clip(np.asarray(prior_values, dtype=np.float64), eps, 1.0-eps)
    counts = np.asarray([len(x) for x, _ in clients], dtype=np.float64)
    model = BaselineCNN(input_channels=channels, num_outputs=2).to(device)

    print(f"FedPU | dataset={args.dataset} | rounds={args.rounds} | "
          f"clients={args.n_clients} | seed={args.seed} | alpha={args.alpha:g}")
    print("method_scope=ICML2022_FedPU_binary_specialization")
    print("risk=CE(P,positive)+pu_weight*(CE(U,negative)-prior*CE(P,negative))")
    print("aggregation=sample_count_FedAvg; model_selection=final_round; threshold=argmax")
    print(f"client_priors={priors.tolist()} | client_samples={counts.astype(int).tolist()}")

    for round_index in range(args.rounds):
        lr = args.lr * (0.995 ** round_index)
        local_models, diagnostics = [], []
        for client_index, ((x, t), prior) in enumerate(zip(clients, priors)):
            local, diag = local_train_fedpu(
                clone_model(model), x, t, prior,
                epochs=args.local_epochs, lr=lr,
                batch_size=args.batch_size, pu_weight=args.pu_weight,
                max_steps_per_epoch=args.max_steps_per_epoch,
                seed=args.seed * 100000 + round_index * args.n_clients + client_index,
            )
            local_models.append(local)
            diagnostics.append(diag)
        average_models(model, local_models, counts)
        acc, recall, auc, f1, threshold = evaluate_two_logit(model, x_test, t_test)
        losses = [round(d["loss"], 6) for d in diagnostics]
        corrections = [round(d["correction"], 6) for d in diagnostics]
        batches = [(d["p_per_batch"], d["u_per_batch"]) for d in diagnostics]
        print(f"FedPU-Diag | round={round_index+1:03d} | lr={lr:.8f} | "
              f"loss={losses} | corrected_negative_risk={corrections} | P_U_batch={batches}")
        print(f"Round {round_index+1:03d} | AUC:{auc:.4f} F1:{f1:.4f} "
              f"Acc:{acc:.4f} Recall:{recall:.4f} Thr:{threshold:.4f}")

    acc, recall, auc, f1, threshold = evaluate_two_logit(model, x_test, t_test)
    print(f"Final | AUC:{auc:.4f} F1:{f1:.4f} Acc:{acc:.4f} "
          f"Recall:{recall:.4f} Thr:{threshold:.4f}")
    output = args.output or f"fedpu_{args.dataset}_seed{args.seed}.pt"
    torch.save({
        "method": "FedPU binary specialization", "model": model.state_dict(),
        "round": args.rounds,
        "metrics": {"auc": auc, "f1": f1, "acc": acc,
                    "recall": recall, "threshold": threshold},
        "priors": priors.tolist(),
    }, output)
    print(f"Checkpoint -> {Path(output).resolve()}")


if __name__ == "__main__":
    main()
