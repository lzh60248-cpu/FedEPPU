"""Independent entry point for controlled ablations of Main2."""

import argparse
import os
import random
import sys
import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Configure cuBLAS before importing torch, but only for explicitly requested
# bitwise-deterministic runs; normal seeded runs retain the fast CUDA path.
if "--strict_determinism" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from datasets.dataset import load_dataset
_DATA_MODULE = "datasets.dataset"

import FedEPPU2_ablation as training_engine
from FedEPPU2_ablation import (
    _PU_LOSS_MODULE,
    BetterCifarCNN,
    device,
    parse_train_data,
    train_fedavg_proposed,
)


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
    input_channels: int
    default_output: str


_CONFIGS: Dict[str, DatasetConfig] = {
    "mnist":         DatasetConfig("mnist",        1000, 57976, 0.1, 1024, 1, "result_fedeppu_mnist.png"),
    "fashion_mnist": DatasetConfig("fashion_mnist", 1000, 57976, 0.1, 1024, 1, "result_fedeppu_fashion_mnist.png"),
    "cifar10":       DatasetConfig("cifar10",       1000, 47976, 0.1, 1024, 3, "result_fedeppu_cifar10.png"),
    "chest":         DatasetConfig("chest",         1000,  3960, 0.1,  256, 1, "result_fedeppu_chest.png"),
    "kolektor_sdd2": DatasetConfig("kolektor_sdd2",  120,  1900, 0.1,  256, 3, "result_fedeppu_kolektor_sdd2.png"),
}
_CONFIGS["fashion-mnist"] = _CONFIGS["fashion_mnist"]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DatasetConfig) -> nn.Module:
    return BetterCifarCNN(input_channels=cfg.input_channels).to(device)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_round_result(
    r, acc, rec, auc, f1, threshold=None,
    specificity=None, bacc=None, fp=None, fn=None,
) -> None:
    line = f"Round {r+1:03d} | AUC:{auc:.4f} F1:{f1:.4f} Acc:{acc:.4f} Recall:{rec:.4f}"
    if specificity is not None and bacc is not None:
        line += f" Spec:{specificity:.4f} BAcc:{bacc:.4f} FP:{fp} FN:{fn}"
    if threshold is not None:
        line += f" Thr:{threshold:.4f}"
    print(line)


def save_curve(history: Dict[str, List[float]], output_path: str, title: str) -> None:
    plt.figure(figsize=(10, 4))
    for key, label in [
        ("auc", "AUC"), ("f1", "F1"), ("acc", "Acc"),
        ("recall", "Recall"), ("specificity", "Specificity"),
        ("bacc", "BAcc"), ("proxy_auc", "Proxy AUC"),
    ]:
        if key in history:
            plt.plot(history[key], label=label)
    plt.title(title)
    plt.xlabel("Round"); plt.ylabel("Metric"); plt.legend()
    plt.tight_layout(); plt.savefig(output_path); plt.close()


def seed_everything(seed: int, strict: bool = False) -> None:
    """Always fix RNGs; enable slower deterministic CUDA kernels only on request."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not strict
    torch.backends.cudnn.deterministic = strict
    torch.use_deterministic_algorithms(strict)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Main2 controlled ablation evaluation.")

    p.add_argument("--dataset",   default="cifar10", choices=list(_CONFIGS.keys()))
    p.add_argument("--rounds",    type=int, default=200)
    p.add_argument("--n_clients", type=int, default=3)
    p.add_argument("--output",    type=str, default=None)
    p.add_argument(
        "--variant", required=True,
        choices=("main2_full", "no_ranking", "fedavg"),
    )
    p.add_argument("--chest_root", type=str, default=None,
                   help="Directory containing Chest train/ and test/ folders.")
    p.add_argument(
        "--chest_val_root", type=str, default=None,
        help="Directory containing Chest validation NORMAL/ and PNEUMONIA/ folders.",
    )
    p.add_argument("--kolektor_root", type=str, default=None,
                   help="Directory containing KolektorSDD2 train/ and test/ folders.")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed for data partition, initialization and training.")
    p.add_argument("--alpha",     type=float, default=None,
                   help="Dirichlet concentration; default is the dataset setting (0.1).")
    p.add_argument("--strict_determinism", action="store_true",
                   help="Use slower deterministic CUDA algorithms for bitwise reruns.")

    a = p.add_argument_group("algorithm")
    a.add_argument("--batch_size", type=int, default=256)
    a.add_argument(
        "--max_steps_per_epoch", type=int, default=15,
        help="Maximum mini-batch optimizer steps in each local epoch.",
    )
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
    if (args.batch_size < 4 or args.rounds < 1 or args.max_steps_per_epoch < 1 or
            args.local_epochs < 1):
        raise ValueError(
            "Batch size must be >= 4; rounds, epochs and steps must be positive."
        )
    seed_everything(args.seed, strict=args.strict_determinism)
    cfg  = _CONFIGS[args.dataset]
    out  = args.output or cfg.default_output
    alpha = cfg.alpha if args.alpha is None else args.alpha

    print(
        f"FedEPPU-Main2-Ablation | variant={args.variant} | dataset={cfg.dataset_name} | rounds={args.rounds} | "
        f"n_clients={args.n_clients} | seed={args.seed} | alpha={alpha:g} | "
        f"strict_determinism={args.strict_determinism}"
    )
    print(f"data_module={_DATA_MODULE}")
    print(f"pu_loss_module={_PU_LOSS_MODULE}")

    print(f"training_module={training_engine.__file__}; ranking=enabled")
    source_root = Path(__file__).resolve().parent
    source_hashes = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("main2_ablation.py", "FedEPPU2_ablation.py", "pu_loss.py", "datasets/dataset.py")
    }
    print("source_sha256=" + json.dumps(source_hashes, sort_keys=True))
    print(f"torch_version={torch.__version__}; cuda_version={torch.version.cuda}; device={device}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"alpha": alpha, "n_proxy": cfg.n_proxy}
    if cfg.dataset_name == "chest":
        if not args.chest_root:
            raise ValueError("--chest_root is required when --dataset chest is used.")
        if not args.chest_val_root:
            raise ValueError(
                "--chest_val_root is required when --dataset chest is used."
            )
        load_kwargs["chest_root"] = args.chest_root
        load_kwargs["chest_val_root"] = args.chest_val_root
    elif cfg.dataset_name == "kolektor_sdd2":
        if not args.kolektor_root:
            raise ValueError("--kolektor_root is required when --dataset kolektor_sdd2 is used.")
        load_kwargs["kolektor_root"] = args.kolektor_root
    c_raw, te_raw, priors, p_raw = load_dataset(
        cfg.dataset_name, cfg.n_labeled, cfg.n_unlabeled,
        args.n_clients, **load_kwargs,
    )
    client_data    = [parse_train_data(d) for d in c_raw]
    x_test, t_test = parse_train_data(te_raw)
    proxy_x, proxy_t = parse_train_data(p_raw)

    if cfg.dataset_name == "chest":
        # The 64x64 unit-range pipeline is intentionally retained: on the
        # observed Chest split it generalizes substantially better than the
        # experimental 96x96 per-image-z-score residual model.
        print("chest_preprocessing=unit_range_64; model=BetterCifarCNN")

    selected_model, history, final_acc = train_fedavg_proposed(
        client_data=client_data, x_test=x_test, t_test=t_test,
        proxy_x=proxy_x, priors=priors,
        anchor_model=build_model(cfg),
        proxy_t=proxy_t,
        rounds=args.rounds, n_clients=args.n_clients,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        local_epochs=args.local_epochs,
        use_amp=not args.no_amp,
        round_callback=print_round_result,
        ablation_variant=args.variant,
    )

    artifact_base = Path(out).with_suffix("")
    metadata = {
        "method": "FedEPPU-Main2-Ablation", "variant": args.variant,
        "ranking_enabled": args.variant != "no_ranking",
        "complement_mass_aggregation": args.variant != "fedavg",
        "negative_support_aggregation": False,
        "model_ema": False,
        "gradient_balance": False, "local_support_scaling": False,
        "args": vars(args), "source_sha256": source_hashes,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "client_priors": [float(p) for p in priors],
        "client_sample_counts": [len(x) for x, _ in client_data],
    }
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in selected_model.state_dict().items()},
        "selected": history["selected"], "metadata": metadata,
    }, str(artifact_base) + ".pt")
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else None
        return value
    with open(str(artifact_base) + ".json", "w", encoding="utf-8") as handle:
        json.dump(json_safe({"metadata": metadata, "history": history}), handle, indent=2, allow_nan=False)
    save_curve(history, out, f"FedEPPU Main2 ablation - {args.variant} - {cfg.dataset_name}")
    print(f"Model/history -> {artifact_base}.pt / {artifact_base}.json")
    print(f"\nCurve → {out}")

    final = history.get("selected")
    if final is not None:
        print("=" * 55)
        print(f"Selected round: {final['round']}")
        print(f"Proxy AUC     : {final['proxy_auc']:.4f}")
        print(f"Max Proxy AUC : {final['proxy_auc_max']:.4f}")
        objective = final.get("threshold_objective", "acc").upper()
        print(f"Proxy {objective:<7}: {final['proxy_score']:.4f}")
        print(f"Proxy Spec    : {final['proxy_specificity']:.4f}")
        print(f"Final Acc     : {final['acc']:.4f}")
        print(f"Final Recall  : {final['recall']:.4f}")
        print(f"Final Spec    : {final['specificity']:.4f}")
        print(f"Final BAcc    : {final['bacc']:.4f}")
        print(f"Final AUC     : {final['auc']:.4f}")
        print(f"Final F1      : {final['f1']:.4f}")
        print(f"Final FP/FN   : {final['fp']}/{final['fn']}")
        print(f"Final Thr     : {final['threshold']:.4f}")
        print("=" * 55)


if __name__ == "__main__":
    main()
