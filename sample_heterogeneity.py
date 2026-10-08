"""Controlled CIFAR-10 client sample-count heterogeneity experiment."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import FedEPPU_ablation as engine
from datasets.dataset import binarize_labels, get_cifar10


SIZE_RATIOS = {
    "balanced": (1, 1, 1),
    "moderate": (6, 3, 1),
    "extreme": (8, 1, 1),
}
AGGREGATIONS = {
    "fedavg": "fedavg",
    "complement": "no_support",
    "support": "no_complement",
    "joint": "full",
}
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


def _allocate(total, ratio):
    ratio = np.asarray(ratio, dtype=np.int64)
    if total % int(ratio.sum()):
        raise ValueError("controlled totals must be divisible by ratio sum")
    return (ratio * (total // int(ratio.sum()))).astype(np.int64)


def controlled_split(size_ratio):
    """Change client sizes while preserving samples, priors and global mass."""
    (x_train, y_train), (x_test, y_test) = get_cifar10()
    y_train, y_test = binarize_labels(y_train, y_test, "cifar10")
    positive = np.random.permutation(np.flatnonzero(y_train == 1))
    negative = np.random.permutation(np.flatnonzero(y_train == -1))
    proxy_idx = np.concatenate((positive[:PROXY_PER_CLASS],
                                negative[:PROXY_PER_CLASS]))
    np.random.shuffle(proxy_idx)
    positive = positive[PROXY_PER_CLASS:]
    negative = negative[PROXY_PER_CLASS:]

    labeled = positive[:TOTAL_LABELED]
    u_positive = positive[TOTAL_LABELED:
                          TOTAL_LABELED + TOTAL_UNLABELED_POSITIVE]
    u_negative = negative[:TOTAL_UNLABELED_NEGATIVE]
    if (len(labeled) != TOTAL_LABELED or
            len(u_positive) != TOTAL_UNLABELED_POSITIVE or
            len(u_negative) != TOTAL_UNLABELED_NEGATIVE):
        raise RuntimeError("controlled CIFAR-10 pool is too small")

    p_counts = _allocate(TOTAL_LABELED, size_ratio)
    up_counts = _allocate(TOTAL_UNLABELED_POSITIVE, size_ratio)
    un_counts = _allocate(TOTAL_UNLABELED_NEGATIVE, size_ratio)
    clients, priors = [], []
    p_cursor = up_cursor = un_cursor = 0
    for p_count, up_count, un_count in zip(p_counts, up_counts, un_counts):
        p_idx = labeled[p_cursor:p_cursor+p_count]
        up_idx = u_positive[up_cursor:up_cursor+up_count]
        un_idx = u_negative[un_cursor:un_cursor+un_count]
        p_cursor += int(p_count)
        up_cursor += int(up_count)
        un_cursor += int(un_count)
        u_idx = np.concatenate((up_idx, un_idx))
        np.random.shuffle(u_idx)
        indices = np.concatenate((p_idx, u_idx))
        observed = np.concatenate((np.ones(len(p_idx), np.int32),
                                   -np.ones(len(u_idx), np.int32)))
        order = np.random.permutation(len(indices))
        clients.append((x_train[indices][order], observed[order]))
        priors.append(float(len(up_idx) / len(u_idx)))

    if (p_cursor != TOTAL_LABELED or
            up_cursor != TOTAL_UNLABELED_POSITIVE or
            un_cursor != TOTAL_UNLABELED_NEGATIVE):
        raise RuntimeError("controlled allocation did not consume each pool")
    proxy = (x_train[proxy_idx], y_train[proxy_idx])
    return clients, (x_test, y_test), proxy, priors


def to_device(pair):
    x, y = pair
    return (torch.from_numpy(np.asarray(x, np.float32)).to(engine.device),
            torch.from_numpy(np.asarray(y, np.int32)).to(engine.device))


def print_round_result(round_index, acc, recall, auc, f1, threshold=None,
                       specificity=None, bacc=None, fp=None, fn=None):
    line = (f"Round {round_index+1:03d} | AUC:{auc:.4f} F1:{f1:.4f} "
            f"Acc:{acc:.4f} Recall:{recall:.4f}")
    if specificity is not None and bacc is not None:
        line += (f" Spec:{specificity:.4f} BAcc:{bacc:.4f} "
                 f"FP:{int(fp)} FN:{int(fn)}")
    if threshold is not None:
        line += f" Thr:{threshold:.4f}"
    print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_config", choices=tuple(SIZE_RATIOS), required=True)
    parser.add_argument("--aggregation", choices=tuple(AGGREGATIONS), required=True)
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
    x_test, y_test = to_device(raw_test)
    proxy_x, proxy_y = to_device(raw_proxy)
    counts = np.asarray([len(x) for x, _ in clients], dtype=np.float64)
    raw_weights = counts / counts.sum()
    complement_weights = engine.ComplementMassAggregation.weights(counts)
    support = np.asarray([
        engine.linear_ranking_support(p, args.batch_size - args.batch_size//4)
        for p in priors
    ])
    selected_base = (complement_weights if args.aggregation in
                     ("complement", "joint") else raw_weights)
    expected_weights = (
        engine.ComplementMassAggregation.support_aware_weights(selected_base, support)
        if args.aggregation in ("support", "joint") else selected_base
    )
    print(f"method=fedeppu; size_config={args.size_config}; ratio={ratio}; "
          f"aggregation={args.aggregation}; seed={args.seed}")
    print(f"client_counts={counts.astype(int).tolist()}; priors={priors}")
    print(f"raw_weights={raw_weights.tolist()}")
    print(f"complement_weights={complement_weights.tolist()}")
    print(f"theoretical_support={support.tolist()}")
    print(f"expected_effective_weights={expected_weights.tolist()}")

    anchor = engine.BetterCifarCNN(input_channels=3).to(engine.device)
    model, history, _ = engine.train_fedavg_proposed(
        client_data=clients, x_test=x_test, t_test=y_test,
        proxy_x=proxy_x, proxy_t=proxy_y, priors=priors,
        anchor_model=anchor, rounds=args.rounds, n_clients=3,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        local_epochs=args.local_epochs, use_amp=True,
        round_callback=print_round_result,
        ablation_variant=AGGREGATIONS[args.aggregation],
    )
    selected = history["selected"]
    final_diag = history["client_diag"][-1]
    metrics = {
        "bacc": selected["bacc"], "auc": selected["auc"],
        "threshold": selected["threshold"], "selected_round": selected["round"],
        "base_weight": [d["base_aggregation_weight"] for d in final_diag],
        "support": [d["support_aggregation_factor"] for d in final_diag],
        "effective_weight": [d["effective_aggregation_weight"] for d in final_diag],
    }
    payload = {
        "method": "fedeppu", "size_config": args.size_config,
        "ratio": list(ratio), "aggregation": args.aggregation,
        "seed": args.seed, "client_counts": counts.astype(int).tolist(),
        "priors": priors, "metrics": metrics,
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
