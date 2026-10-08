"""CIFAR-10 FedAvg-nnPU under four Dirichlet concentrations."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

import FedEPPU3
from other.baseline_runtime import BaselineCNN, device, evaluate_metrics
from datasets.dataset import load_dataset
from other.fedavg_nnpu import local_train_fedavg
from other.nnpu_common import average_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--local_epochs", type=int, default=6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def print_metrics(round_index, metrics):
    print(
        f"Round {round_index + 1:03d} | AUC:{metrics['auc']:.4f} "
        f"F1:{metrics['f1']:.4f} Acc:{metrics['acc']:.4f} "
        f"BAcc:{metrics['bacc']:.4f} Recall:{metrics['recall']:.4f} Thr:0.0000",
        flush=True,
    )


def main():
    args = parse_args()
    if args.alpha <= 0 or min(args.rounds, args.n_clients, args.batch_size,
                               args.local_epochs, args.max_steps_per_epoch) < 1:
        raise ValueError("alpha, rounds, clients, batch size, epochs and step cap must be positive")
    seed_all(args.seed)
    raw_clients, raw_test, priors, _ = load_dataset(
        "cifar10", 1000, 47976, args.n_clients, alpha=args.alpha, n_proxy=1024
    )
    clients = [FedEPPU3.parse_train_data(records) for records in raw_clients]
    priors = np.clip(np.asarray(priors, float),
                     np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    sample_counts = [int(len(x)) for x, _ in clients]
    x_test, t_test = FedEPPU3.parse_train_data(raw_test)
    model = BaselineCNN(3, 1).to(device)
    weights = np.asarray(sample_counts, dtype=float)
    history = []

    print(
        f"method=FedAvg-nnPU; dataset=cifar10; alpha={args.alpha:g}; seed={args.seed}; "
        f"clients={args.n_clients}; rounds={args.rounds}"
    )
    print("client_priors=" + ",".join(f"{value:.6f}" for value in priors))
    print("client_sample_counts=" + ",".join(str(value) for value in sample_counts))

    for round_index in range(args.rounds):
        local_models = []
        for client_index, (x, labels) in enumerate(clients):
            local, _ = local_train_fedavg(
                copy.deepcopy(model), x, labels, float(priors[client_index]),
                args.local_epochs, args.lr, args.batch_size,
                max_steps_per_epoch=args.max_steps_per_epoch,
            )
            local_models.append(local)
        average_models(model, local_models, weights)
        metrics = evaluate_metrics(model, x_test, t_test)
        history.append({key: float(value) for key, value in metrics.items()})
        print_metrics(round_index, metrics)

    auc = np.asarray([row["auc"] for row in history], dtype=float)
    payload = {
        "method": "fedavg_nnpu",
        "dataset": "cifar10",
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "client_priors": priors.tolist(),
        "client_sample_counts": sample_counts,
        "metrics": history[-1],
        "history": history,
        "last20_auc_mean": float(auc[-20:].mean()),
        "last20_auc_sd": float(auc[-20:].std(ddof=1)),
        "peak_auc": float(auc.max()),
        "peak_to_last20_auc_drop": float(auc.max() - auc[-20:].mean()),
        "protocol": "final_round+fixed_logit_0+no_proxy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **payload}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
