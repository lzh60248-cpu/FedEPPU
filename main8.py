"""Chest entry point cloned from Main5's D9 ablation."""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from datasets.dataset8 import load_dataset
_DATA_MODULE = "datasets.dataset8"

try:
    from model.model import CNNMnist
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "main8.py requires model/model.py with CNNMnist."
    ) from e

from FedEPPU8 import (
    BetterCifarCNN,
    device,
    parse_train_data,
    train_fedavg_proposed,
)

EXPERIMENT_ID = "main8_from_main5_chest_v1"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    dataset_name:   str
    n_labeled:      int
    n_unlabeled:    int
    alpha:          float
    n_proxy:        int
    model_type:     str
    input_channels: int


_CONFIGS: Dict[str, DatasetConfig] = {
    "mnist":         DatasetConfig("mnist",        1000, 57976, 0.1, 1024, "mnist_cnn",  1),
    "fashion_mnist": DatasetConfig("fashion_mnist", 1000, 57976, 0.1, 1024, "mnist_cnn",  1),
    "cifar10":       DatasetConfig("cifar10",       1000, 47976, 0.1, 1024, "better_cnn", 3),
    "chest":         DatasetConfig("chest",         1000,  3960, 0.1,  256, "better_cnn", 1),
}
_CONFIGS["fashion-mnist"] = _CONFIGS["fashion_mnist"]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DatasetConfig) -> nn.Module:
    if cfg.model_type == "better_cnn":
        return BetterCifarCNN(input_channels=cfg.input_channels).to(device)
    model = CNNMnist().to(device)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in ("fc2", "fc", "classifier"):
            setattr(model, name, nn.Linear(module.in_features, 1).to(device))
    return model


# ---------------------------------------------------------------------------
# Reproducible fast path
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Fix RNG seeds while retaining the fast CUDA execution path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FedEPPU evaluation.")

    p.add_argument("--dataset",   default="chest", choices=["chest"])
    p.add_argument("--rounds",    type=int, default=200)
    p.add_argument("--n_clients", type=int, default=3)
    p.add_argument("--chest_root", type=str, default=None,
                   help="Directory containing Chest train/ and test/ folders.")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed for data partition, initialization and training.")
    p.add_argument("--alpha",     type=float, default=None,
                   help="Dirichlet concentration; default is the dataset setting (0.1).")

    a = p.add_argument_group("algorithm")
    a.add_argument(
        "--n_shrink", type=float, default=200.0,
        help="Empirical-Bayes prior shrinkage sample size.",
    )
    a.add_argument(
        "--lambda_prior", type=float, default=0.10,
        help="Base weight of client U-label-distribution consistency.",
    )
    a.add_argument("--batch_size", type=int, default=256)
    a.add_argument("--max_steps_per_epoch", type=int, default=128)
    a.add_argument("--local_epochs", type=int, default=6)
    a.add_argument("--no_amp", action="store_true",
                   help="Disable CUDA automatic mixed precision.")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    if args.alpha is not None and args.alpha <= 0:
        raise ValueError("--alpha must be greater than zero.")
    if args.lambda_prior < 0 or args.n_shrink < 0:
        raise ValueError("--lambda_prior and --n_shrink must be non-negative.")
    if args.batch_size < 1 or args.max_steps_per_epoch < 1 or args.local_epochs < 1:
        raise ValueError(
            "--batch_size, --max_steps_per_epoch and --local_epochs must be positive."
        )
    seed_everything(args.seed)
    cfg  = _CONFIGS[args.dataset]
    alpha = cfg.alpha if args.alpha is None else args.alpha

    print(f"experiment={EXPERIMENT_ID}")
    print(
        f"FedEPPU8-FromMain5-D9-NoModelEMA | dataset={cfg.dataset_name} | rounds={args.rounds} | "
        f"n_clients={args.n_clients} | seed={args.seed} | alpha={alpha:g}"
    )
    print(f"data_module={_DATA_MODULE}")

    load_kwargs = {"alpha": alpha, "n_proxy": cfg.n_proxy}
    if cfg.dataset_name == "chest":
        if not args.chest_root:
            raise ValueError("--chest_root is required when --dataset chest is used.")
        load_kwargs["chest_root"] = args.chest_root

    c_raw, te_raw, priors, p_raw = load_dataset(
        cfg.dataset_name, cfg.n_labeled, cfg.n_unlabeled,
        args.n_clients, **load_kwargs,
    )
    client_data    = [parse_train_data(d) for d in c_raw]
    x_test, t_test = parse_train_data(te_raw)
    proxy_x, proxy_t = parse_train_data(p_raw)
    print(
        f"proxy_sampling=main5_random_train_split; "
        f"positive={int((proxy_t > 0).sum().item())}; "
        f"negative={int((proxy_t < 0).sum().item())}"
    )

    _, history, _ = train_fedavg_proposed(
        client_data=client_data, x_test=x_test, t_test=t_test,
        proxy_x=proxy_x, priors=priors,
        anchor_model=build_model(cfg),
        proxy_t=proxy_t,
        rounds=args.rounds, n_clients=args.n_clients,
        n_shrink=args.n_shrink,
        lambda_prior=args.lambda_prior,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        local_epochs=args.local_epochs,
        use_amp=False,
    )

    final = history.get("selected")
    if final is not None:
        print("=" * 55)
        print(f"Selected round: {final['round']}")
        print(f"Proxy AUC     : {final['proxy_auc']:.4f}")
        print(f"Final Acc     : {final['acc']:.4f}")
        print(f"Final Recall  : {final['recall']:.4f}")
        print(f"Final AUC     : {final['auc']:.4f}")
        print(f"Final F1      : {final['f1']:.4f}")
        print(f"Final Thr     : {final['threshold']:.4f}")
        print("=" * 55)


if __name__ == "__main__":
    main()
