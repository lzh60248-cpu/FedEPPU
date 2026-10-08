"""Isolate client-prior heterogeneity at equal client sample counts."""

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


PRIOR_CONFIGS = {
    "mild": (0.30, 0.50, 0.70),
    "moderate": (0.10, 0.50, 0.90),
    "extreme": (0.01, 0.50, 0.99),
}
METHODS = ("fedavg_nnpu", "ranking_fedavg", "fedeppu")
P_PER_CLIENT = 333
U_PER_CLIENT = 15640
PROXY_PER_CLASS = 512


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def controlled_split(priors):
    """Keep all client sizes and global class totals fixed across settings."""
    (x_train, y_train), (x_test, y_test) = get_cifar10()
    y_train, y_test = binarize_labels(y_train, y_test, "cifar10")
    positive = np.random.permutation(np.flatnonzero(y_train == 1))
    negative = np.random.permutation(np.flatnonzero(y_train == -1))

    proxy_idx = np.concatenate((positive[:PROXY_PER_CLASS], negative[:PROXY_PER_CLASS]))
    np.random.shuffle(proxy_idx)
    positive = positive[PROXY_PER_CLASS:]
    negative = negative[PROXY_PER_CLASS:]

    labelled_total = 3 * P_PER_CLIENT
    labelled = positive[:labelled_total]
    positive = positive[labelled_total:]
    total_u_positive = int(round(sum(priors) * U_PER_CLIENT))
    total_u_negative = 3 * U_PER_CLIENT - total_u_positive
    if len(positive) < total_u_positive or len(negative) < total_u_negative:
        raise RuntimeError("Controlled CIFAR-10 pool is too small.")
    u_positive = positive[:total_u_positive]
    u_negative = negative[:total_u_negative]

    clients = []
    pos_cursor = neg_cursor = 0
    for client_index, prior in enumerate(priors):
        n_u_pos = int(round(prior * U_PER_CLIENT))
        n_u_neg = U_PER_CLIENT - n_u_pos
        p_idx = labelled[
            client_index * P_PER_CLIENT:(client_index + 1) * P_PER_CLIENT
        ]
        u_idx = np.concatenate((
            u_positive[pos_cursor:pos_cursor + n_u_pos],
            u_negative[neg_cursor:neg_cursor + n_u_neg],
        ))
        pos_cursor += n_u_pos
        neg_cursor += n_u_neg
        np.random.shuffle(u_idx)
        indices = np.concatenate((p_idx, u_idx))
        observed = np.concatenate((
            np.ones(P_PER_CLIENT, dtype=np.int32),
            -np.ones(U_PER_CLIENT, dtype=np.int32),
        ))
        order = np.random.permutation(len(indices))
        clients.append((x_train[indices][order], observed[order]))
    if pos_cursor != total_u_positive or neg_cursor != total_u_negative:
        raise RuntimeError("Controlled allocation did not preserve global totals.")
    return clients, (x_test, y_test), (x_train[proxy_idx], y_train[proxy_idx])


def to_device(pair, target_device):
    x, y = pair
    return (
        torch.from_numpy(np.asarray(x, dtype=np.float32)).to(target_device),
        torch.from_numpy(np.asarray(y, dtype=np.int32)).to(target_device),
    )


def print_round(round_index, acc, recall, auc, f1, threshold=None,
                specificity=None, bacc=None, fp=None, fn=None):
    print(
        f"Round {round_index + 1:03d} | AUC:{auc:.4f} F1:{f1:.4f} "
        f"Acc:{acc:.4f} BAcc:{bacc:.4f} Recall:{recall:.4f} "
        f"Thr:{threshold:.4f}", flush=True,
    )


def run_fedeppu(args, clients, priors, test, proxy, use_fedavg):
    x_test, y_test = test
    proxy_x, proxy_y = proxy
    if use_fedavg:
        engine.ComplementMassAggregation.weights = staticmethod(
            lambda counts: np.asarray(counts, dtype=np.float64) /
            np.asarray(counts, dtype=np.float64).sum()
        )
        engine.ComplementMassAggregation.support_aware_weights = staticmethod(
            lambda base, support: np.asarray(base, dtype=np.float64).copy()
        )
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
        "base_weights": [float(item["base_aggregation_weight"]) for item in final_diag],
        "effective_weights": [float(item["effective_aggregation_weight"]) for item in final_diag],
    }
    return model, metrics, curves, diagnostics


def run_fedavg(args, clients, priors, test, proxy):
    x_test, y_test = test
    proxy_x, proxy_y = proxy
    (selection_x, selection_y), (calibration_x, calibration_y) = split_proxy(
        proxy_x, proxy_y
    )
    model = engine.BetterCifarCNN(input_channels=3).to(engine.device)
    counts = np.asarray([len(x) for x, _ in clients], dtype=np.float64)
    weights = counts / counts.sum()
    history = []
    selector = ProxySelector(interval=5, tolerance=0.001)
    for round_index in range(args.rounds):
        local_models = []
        for client_index, (x, labels) in enumerate(clients):
            local = copy.deepcopy(model)
            local, _ = local_train_fedavg(
                local, x, labels, priors[client_index],
                args.local_epochs, 0.0015, args.batch_size,
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
    return model, selected, curves, {"aggregation_weights": weights.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--prior_config", choices=tuple(PRIOR_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--local_epochs", type=int, default=6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_all(args.seed)
    priors = PRIOR_CONFIGS[args.prior_config]
    raw_clients, raw_test, raw_proxy = controlled_split(priors)
    clients = [to_device(pair, engine.device) for pair in raw_clients]
    test = to_device(raw_test, engine.device)
    proxy = to_device(raw_proxy, engine.device)
    total_u_positive = int(round(sum(priors) * U_PER_CLIENT))
    total_u_negative = 3 * U_PER_CLIENT - total_u_positive
    print(
        f"method={args.method}; config={args.prior_config}; priors={priors}; "
        f"seed={args.seed}; engine={'FedEPPU_rankgate.py' if args.method != 'fedavg_nnpu' else 'FedAvg-nnPU'}"
    )
    print(f"client_sizes={[len(x) for x, _ in clients]}; P={P_PER_CLIENT}; U={U_PER_CLIENT}")
    print(f"global_U_positive={total_u_positive}; global_U_negative={total_u_negative}")

    if args.method in ("fedeppu", "ranking_fedavg"):
        model, metrics, history, diagnostics = run_fedeppu(
            args, clients, priors, test, proxy,
            use_fedavg=args.method == "ranking_fedavg",
        )
    else:
        model, metrics, history, diagnostics = run_fedavg(
            args, clients, priors, test, proxy
        )
    payload = {
        "method": args.method, "prior_config": args.prior_config,
        "priors": list(priors), "seed": args.seed,
        "p_per_client": P_PER_CLIENT, "u_per_client": U_PER_CLIENT,
        "global_u_positive": total_u_positive,
        "global_u_negative": total_u_negative,
        "expected_negative_counts": [
            (1.0 - float(prior)) * (args.batch_size - args.batch_size // 4)
            for prior in priors
        ],
        "support_factors": [
            min(1.0, (1.0 - float(prior)) *
                (args.batch_size - args.batch_size // 4))
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
