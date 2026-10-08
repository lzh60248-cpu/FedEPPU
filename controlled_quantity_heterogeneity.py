"""Isolate client quantity heterogeneity at fixed client class prior."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

import FedEPPU_rankgate as engine
from datasets.dataset import binarize_labels, get_cifar10
from other.fedavg_nnpu import local_train_fedavg
from other.nnpu_common import average_models
from other.proxy_protocol import ProxySelector, calibrated_threshold, metrics, model_scores, split_proxy


SIZE_RATIOS = {
    "balanced": (1, 1, 1),
    "moderate": (6, 3, 1),
    "extreme": (8, 1, 1),
}
METHODS = ("fedavg_nnpu", "complement_nnpu", "fedeppu")
TOTAL_LABELED = 990
TOTAL_UNLABELED_POSITIVE = 22920
TOTAL_UNLABELED_NEGATIVE = 24000
PROXY_PER_CLASS = 512


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def allocate(total, ratio):
    ratio = np.asarray(ratio, dtype=np.int64)
    if total % int(ratio.sum()):
        raise ValueError("Controlled total must be divisible by the ratio sum.")
    return ratio * (total // int(ratio.sum()))


def controlled_split(size_ratio):
    """Change only client sample counts while preserving priors and global pools."""
    (x_train, y_train), (x_test, y_test) = get_cifar10()
    y_train, y_test = binarize_labels(y_train, y_test, "cifar10")
    positive = np.random.permutation(np.flatnonzero(y_train == 1))
    negative = np.random.permutation(np.flatnonzero(y_train == -1))
    proxy_idx = np.concatenate((positive[:PROXY_PER_CLASS], negative[:PROXY_PER_CLASS]))
    np.random.shuffle(proxy_idx)
    positive = positive[PROXY_PER_CLASS:]
    negative = negative[PROXY_PER_CLASS:]

    labeled = positive[:TOTAL_LABELED]
    u_positive = positive[TOTAL_LABELED:TOTAL_LABELED + TOTAL_UNLABELED_POSITIVE]
    u_negative = negative[:TOTAL_UNLABELED_NEGATIVE]
    if (len(labeled) != TOTAL_LABELED or
            len(u_positive) != TOTAL_UNLABELED_POSITIVE or
            len(u_negative) != TOTAL_UNLABELED_NEGATIVE):
        raise RuntimeError("Controlled CIFAR-10 pool is too small.")

    p_counts = allocate(TOTAL_LABELED, size_ratio)
    up_counts = allocate(TOTAL_UNLABELED_POSITIVE, size_ratio)
    un_counts = allocate(TOTAL_UNLABELED_NEGATIVE, size_ratio)
    clients, priors = [], []
    p_cursor = up_cursor = un_cursor = 0
    for p_count, up_count, un_count in zip(p_counts, up_counts, un_counts):
        p_idx = labeled[p_cursor:p_cursor + p_count]
        up_idx = u_positive[up_cursor:up_cursor + up_count]
        un_idx = u_negative[un_cursor:un_cursor + un_count]
        p_cursor += int(p_count)
        up_cursor += int(up_count)
        un_cursor += int(un_count)
        u_idx = np.concatenate((up_idx, un_idx))
        np.random.shuffle(u_idx)
        indices = np.concatenate((p_idx, u_idx))
        observed = np.concatenate((
            np.ones(len(p_idx), dtype=np.int32),
            -np.ones(len(u_idx), dtype=np.int32),
        ))
        order = np.random.permutation(len(indices))
        clients.append((x_train[indices][order], observed[order]))
        priors.append(float(len(up_idx) / len(u_idx)))
    if (p_cursor != TOTAL_LABELED or up_cursor != TOTAL_UNLABELED_POSITIVE or
            un_cursor != TOTAL_UNLABELED_NEGATIVE):
        raise RuntimeError("Controlled allocation did not consume each pool.")
    return clients, (x_test, y_test), (x_train[proxy_idx], y_train[proxy_idx]), priors


def to_device(pair):
    x, y = pair
    return (
        torch.from_numpy(np.asarray(x, dtype=np.float32)).to(engine.device),
        torch.from_numpy(np.asarray(y, dtype=np.int32)).to(engine.device),
    )


def print_round(round_index, acc, recall, auc, f1, threshold=None,
                specificity=None, bacc=None, fp=None, fn=None):
    print(
        f"Round {round_index + 1:03d} | AUC:{auc:.4f} F1:{f1:.4f} "
        f"Acc:{acc:.4f} BAcc:{bacc:.4f} Recall:{recall:.4f} "
        f"Thr:{threshold:.4f}", flush=True,
    )


def run_fedeppu(args, clients, priors, test, proxy):
    x_test, y_test = test
    proxy_x, proxy_y = proxy
    model, history, _ = engine.train_fedavg_proposed(
        client_data=clients, x_test=x_test, t_test=y_test,
        proxy_x=proxy_x, proxy_t=proxy_y, priors=priors,
        anchor_model=engine.BetterCifarCNN(input_channels=3).to(engine.device),
        rounds=args.rounds, n_clients=3, batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        local_epochs=args.local_epochs, use_amp=True, ranking_gate="hard",
        round_callback=print_round,
    )
    selected = history["selected"]
    metrics = {key: float(selected[key]) for key in ("bacc", "acc", "auc", "f1")}
    metrics["selected_round"] = int(selected["round"])
    curves = {key: [float(value) for value in history[key]]
              for key in ("bacc", "acc", "auc", "f1")}
    final_diag = history["client_diag"][-1]
    diagnostics = {
        "aggregation_mode": "complement_support",
        "base_weights": [float(item["base_aggregation_weight"]) for item in final_diag],
        "effective_weights": [float(item["effective_aggregation_weight"]) for item in final_diag],
    }
    return model, metrics, curves, diagnostics


def run_nnpu_aggregation(args, clients, priors, test, proxy, use_complement):
    x_test, y_test = test
    proxy_x, proxy_y = proxy
    (selection_x, selection_y), (calibration_x, calibration_y) = split_proxy(
        proxy_x, proxy_y
    )
    model = engine.BetterCifarCNN(input_channels=3).to(engine.device)
    counts = np.asarray([len(x) for x, _ in clients], dtype=np.float64)
    weights = (engine.ComplementMassAggregation.weights(counts)
               if use_complement else counts / counts.sum())
    history = []
    selector = ProxySelector(interval=5, tolerance=0.001)
    for round_index in range(args.rounds):
        local_models = []
        for client_index, (x, labels) in enumerate(clients):
            local = copy.deepcopy(model)
            local, _ = local_train_fedavg(
                local, x, labels, priors[client_index], args.local_epochs,
                0.0015, args.batch_size,
                max_steps_per_epoch=args.max_steps_per_epoch,
            )
            local_models.append(local)
        average_models(model, local_models, weights)
        test_scores = model_scores(model, x_test)
        row = metrics(y_test.detach().cpu().numpy(), test_scores, 0.0)
        proxy_auc = selector.consider(
            round_index, args.rounds, model_scores(model, selection_x),
            selection_y.detach().cpu().numpy(), model.state_dict(),
        )
        row["proxy_auc"] = proxy_auc
        history.append(row)
        print(
            f"Round {round_index + 1:03d} | AUC:{row['auc']:.4f} "
            f"F1:{row['f1']:.4f} Acc:{row['acc']:.4f} "
            f"BAcc:{row['bacc']:.4f} ProxyAUC:{proxy_auc:.4f}", flush=True,
        )
    model.load_state_dict(selector.state)
    threshold = calibrated_threshold(
        calibration_y.detach().cpu().numpy(), model_scores(model, calibration_x)
    )
    selected = metrics(
        y_test.detach().cpu().numpy(), model_scores(model, x_test), threshold
    )
    selected["selected_round"] = selector.selected_round
    selected["selected_proxy_auc"] = selector.selected_auc
    curves = {key: [float(row[key]) for row in history]
              for key in ("bacc", "acc", "auc", "f1")}
    return model, selected, curves, {
        "aggregation_mode": "complement" if use_complement else "sample",
        "aggregation_weights": weights.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_config", choices=tuple(SIZE_RATIOS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--local_epochs", type=int, default=6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_all(args.seed)

    ratio = SIZE_RATIOS[args.size_config]
    raw_clients, raw_test, raw_proxy, priors = controlled_split(ratio)
    clients = [to_device(pair) for pair in raw_clients]
    test = to_device(raw_test)
    proxy = to_device(raw_proxy)
    counts = np.asarray([len(x) for x, _ in clients], dtype=np.float64)
    fedavg_weights = counts / counts.sum()
    complement_weights = engine.ComplementMassAggregation.weights(counts)
    print(
        f"experiment=4.3.2; method={args.method}; size_config={args.size_config}; "
        f"ratio={ratio}; seed={args.seed}"
    )
    print(f"client_counts={counts.astype(int).tolist()}; priors={priors}")
    print(f"fedavg_weights={fedavg_weights.tolist()}")
    print(f"complement_weights={complement_weights.tolist()}")

    if args.method in ("fedavg_nnpu", "complement_nnpu"):
        model, metrics, history, diagnostics = run_nnpu_aggregation(
            args, clients, priors, test, proxy,
            use_complement=args.method == "complement_nnpu",
        )
    else:
        model, metrics, history, diagnostics = run_fedeppu(
            args, clients, priors, test, proxy
        )
    payload = {
        "method": args.method, "size_config": args.size_config,
        "ratio": list(ratio), "seed": args.seed,
        "client_counts": counts.astype(int).tolist(), "priors": priors,
        "fedavg_weights": fedavg_weights.tolist(),
        "complement_weights": complement_weights.tolist(),
        "total_labeled": TOTAL_LABELED,
        "total_unlabeled_positive": TOTAL_UNLABELED_POSITIVE,
        "total_unlabeled_negative": TOTAL_UNLABELED_NEGATIVE,
        "expected_negative_counts": [
            (1.0 - float(prior)) * (args.batch_size - args.batch_size // 4)
            for prior in priors
        ],
        "metrics": metrics, "history": history, "diagnostics": diagnostics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **payload}, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
