"""Section 4.2 runner with per-round histories for every independent baseline."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    roc_auc_score,
)

from .baseline_runtime import (
    BaselineCNN, device, evaluate_fixed, evaluate_metrics, evaluate_two_logit,
)
from .centralized import train_centralized_nnpu, train_centralized_pn
from .data_protocol import CONFIGS, load_protocol, pooled_pu_prior
from .fedavg_nnpu import local_train_fedavg
from .fedprox_nnpu import local_train_fedprox
from .fedpu import local_train_fedpu
from .local_nnpu import train_local_models
from .mrfl_nnpu import MRFLState, local_train_mrfl_nnpu, pu_detection_vector
from .nnpu_common import average_models, clone_model
from .scaffold_nnpu import local_train_scaffold, update_server_control, zero_control


METHODS = ("fedavg_nnpu", "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Independent PU baselines")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", default="cifar10", choices=tuple(CONFIGS))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--local_epochs", type=int, default=6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--fedpu_lr", type=float, default=0.01)
    parser.add_argument("--fedpu_weight", type=float, default=1.0)
    parser.add_argument("--mrfl_warmup_rounds", type=int, default=20)
    parser.add_argument("--chest_root", default=None)
    parser.add_argument("--chest_val_root", default=None)
    parser.add_argument("--kolektor_root", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sync_cuda():
    """Make wall-clock measurements valid when CUDA launches asynchronously."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor(array, dtype):
    return torch.from_numpy(np.asarray(array, dtype=dtype)).to(device)


def fixed_metrics(labels, logits):
    labels = np.asarray(labels).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    prediction = np.where(logits >= 0.0, 1, -1)
    try:
        auc = roc_auc_score(labels, logits)
    except ValueError:
        auc = 0.5
    return {
        "auc": float(auc),
        "f1": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
        "acc": float(accuracy_score(labels, prediction)),
        "bacc": float(balanced_accuracy_score(labels, prediction)),
        "recall": float(recall_score(labels, prediction, pos_label=1, zero_division=0)),
        "threshold": 0.0,
    }


def ensemble_metrics(models, weights, x_test, t_test):
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    combined = None
    for weight, model in zip(weights, models):
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(x_test), 512):
                parts.append(model(x_test[start:start + 512]).reshape(-1).cpu())
        logits = torch.cat(parts).numpy()
        combined = weight * logits if combined is None else combined + weight * logits
    return fixed_metrics(t_test.detach().cpu().numpy(), combined)


def print_metrics(prefix, metrics):
    print(
        f"{prefix} | AUC:{metrics['auc']:.4f} F1:{metrics['f1']:.4f} "
        f"Acc:{metrics['acc']:.4f} BAcc:{metrics['bacc']:.4f} "
        f"Recall:{metrics['recall']:.4f} Thr:{metrics['threshold']:.4f}"
    )


def main():
    args = parse_args()
    positive = (args.rounds, args.n_clients, args.batch_size,
                args.local_epochs, args.max_steps_per_epoch)
    if min(positive) < 1 or args.alpha <= 0 or args.lr <= 0:
        raise ValueError("rounds, clients, batch, epochs, steps, alpha and lr must be positive")
    seed_all(args.seed)
    protocol = load_protocol(
        args.dataset, args.n_clients, args.alpha, args.chest_root,
        args.chest_val_root, args.kolektor_root,
    )
    clients = [
        (tensor(client.x, np.float32), tensor(client.observed_y, np.int32))
        for client in protocol.clients
    ]
    truths = [tensor(client.true_y, np.int32) for client in protocol.clients]
    priors = np.asarray([client.prior for client in protocol.clients], float)
    eps = np.finfo(np.float64).eps
    priors = np.clip(priors, eps, 1.0 - eps)
    counts = np.asarray([len(client.x) for client in protocol.clients], float)
    x_test = tensor(protocol.test_x, np.float32)
    t_test = tensor(protocol.test_y, np.int32)
    outputs = 2 if args.method == "fedpu" else 1
    model = BaselineCNN(protocol.input_channels, outputs).to(device)
    print(
        f"Baseline | method={args.method} | dataset={args.dataset} | "
        f"rounds={args.rounds} | seed={args.seed} | alpha={args.alpha:g}"
    )
    print("selection=final_round; threshold=fixed_logit_0; proxy_access=none")
    history = []
    common = dict(epochs=args.local_epochs, lr=args.lr,
                  batch_size=args.batch_size,
                  max_steps_per_epoch=args.max_steps_per_epoch)

    timing = []
    if args.method in METHODS:
        server_control = zero_control(model) if args.method == "scaffold_nnpu" else None
        client_controls = ([zero_control(model) for _ in clients]
                           if server_control is not None else None)
        mrfl = (MRFLState(args.n_clients, args.mrfl_warmup_rounds, args.seed)
                if args.method == "mrfl_nnpu" else None)
        for round_index in range(args.rounds):
            sync_cuda()
            round_start = time.perf_counter()
            local_models = []
            diagnostics = []
            old_controls, new_controls = [], []
            reference = clone_model(model)
            for index, (x, labels) in enumerate(clients):
                local = clone_model(model)
                if args.method == "fedavg_nnpu":
                    local, diag = local_train_fedavg(local, x, labels, priors[index], **common)
                elif args.method == "fedprox_nnpu":
                    local, diag = local_train_fedprox(local, reference, x, labels,
                                                      priors[index], mu=args.mu, **common)
                elif args.method == "fedpu":
                    fedpu_common = dict(common)
                    fedpu_common["lr"] = args.fedpu_lr * (0.995 ** round_index)
                    local, diag = local_train_fedpu(local, x, labels, priors[index],
                                                     pu_weight=args.fedpu_weight,
                                                     seed=(args.seed * 100000 +
                                                           round_index * args.n_clients + index),
                                                     **fedpu_common)
                elif args.method == "scaffold_nnpu":
                    old = {key: value.clone() for key, value in client_controls[index].items()}
                    local, new, diag = local_train_scaffold(
                        local, reference, x, labels, priors[index],
                        client_controls[index], server_control, **common,
                    )
                    old_controls.append(old)
                    new_controls.append(new)
                    client_controls[index] = new
                else:
                    local, diag = local_train_mrfl_nnpu(
                        local, x, labels, priors[index],
                        is_noisy=bool(mrfl.noisy[index]),
                        seed=args.seed + round_index * args.n_clients + index,
                        **common,
                    )
                local_models.append(local)
                diagnostics.append(diag)
            sync_cuda()
            local_train_seconds = time.perf_counter() - round_start
            aggregation_start = time.perf_counter()
            weights = (np.asarray([d["aggregation_mass"] for d in diagnostics], float)
                       if mrfl is not None else counts)
            average_models(model, local_models, weights)
            if server_control is not None:
                update_server_control(server_control, old_controls, new_controls,
                                      args.n_clients)
            if mrfl is not None:
                vectors = np.stack([
                    pu_detection_vector(model, x, labels) for x, labels in clients
                ])
                mrfl.update_client_division(vectors, round_index)
            sync_cuda()
            aggregation_seconds = time.perf_counter() - aggregation_start
            train_aggregate_seconds = time.perf_counter() - round_start
            timing.append({
                "round": round_index + 1,
                "local_train_seconds": local_train_seconds,
                "aggregation_seconds": aggregation_seconds,
                "train_aggregate_seconds": train_aggregate_seconds,
            })
            print(
                f"[Timing] round={round_index + 1}; "
                f"local_train_seconds={local_train_seconds:.6f}; "
                f"aggregation_seconds={aggregation_seconds:.6f}; "
                f"train_aggregate_seconds={train_aggregate_seconds:.6f}"
            )
            metrics = evaluate_metrics(
                model, x_test, t_test, two_logit=args.method == "fedpu"
            )
            history.append(dict(metrics))
            print_metrics(f"Round {round_index + 1:03d}", metrics)
        saved_model = model.state_dict()

    else:
        raise ValueError(f"Timing experiment does not support {args.method}")

    print_metrics("Final", metrics)
    output = Path(args.output or
                  f"baseline_{args.dataset}_{args.method}_seed{args.seed}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"method": args.method, "model": saved_model,
                "round": args.rounds, "metrics": metrics, "history": history,
                "protocol": "final_round+fixed_logit_0+no_proxy",
                "timing": timing}, output)
    output.with_suffix(".json").write_text(json.dumps({
        "method": args.method, "dataset": args.dataset, "seed": args.seed,
                "metrics": metrics, "history": history, "timing": timing,
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "communication_model_equivalents_per_round": int(
            (4 if args.method == "scaffold_nnpu" else 2) * args.n_clients
        ),
    }, indent=2), encoding="utf-8")
    print(f"Output -> {output.resolve()}")


if __name__ == "__main__":
    main()
