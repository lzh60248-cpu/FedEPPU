"""Controlled CIFAR-10 client-prior heterogeneity experiment."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import FedEPPU
from datasets.dataset import binarize_labels, get_cifar10


PRIOR_CONFIGS = {
    "low": (0.4, 0.5, 0.6),
    "medium": (0.2, 0.5, 0.8),
    "extreme": (0.05, 0.5, 0.95),
}
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
    """Keep client sizes and global P/N totals fixed across prior settings."""
    (x_train, y_train), (x_test, y_test) = get_cifar10()
    y_train, y_test = binarize_labels(y_train, y_test, "cifar10")
    positive = np.random.permutation(np.flatnonzero(y_train == 1))
    negative = np.random.permutation(np.flatnonzero(y_train == -1))
    proxy_idx = np.concatenate((positive[:PROXY_PER_CLASS],
                                negative[:PROXY_PER_CLASS]))
    np.random.shuffle(proxy_idx)
    positive = positive[PROXY_PER_CLASS:]
    negative = negative[PROXY_PER_CLASS:]

    labelled_total = 3 * P_PER_CLIENT
    labelled = positive[:labelled_total]
    positive = positive[labelled_total:]
    total_u_positive = int(round(sum(priors) * U_PER_CLIENT))
    total_u_negative = 3 * U_PER_CLIENT - total_u_positive
    if len(positive) < total_u_positive or len(negative) < total_u_negative:
        raise RuntimeError("controlled CIFAR-10 pool is too small")
    u_positive = positive[:total_u_positive]
    u_negative = negative[:total_u_negative]

    clients = []
    p_cursor = n_cursor = 0
    for client, prior in enumerate(priors):
        n_u_pos = int(round(prior * U_PER_CLIENT))
        n_u_neg = U_PER_CLIENT - n_u_pos
        p_idx = labelled[client * P_PER_CLIENT:(client + 1) * P_PER_CLIENT]
        u_idx = np.concatenate((u_positive[p_cursor:p_cursor + n_u_pos],
                                u_negative[n_cursor:n_cursor + n_u_neg]))
        p_cursor += n_u_pos
        n_cursor += n_u_neg
        np.random.shuffle(u_idx)
        indices = np.concatenate((p_idx, u_idx))
        observed = np.concatenate((np.ones(P_PER_CLIENT, np.int32),
                                   -np.ones(U_PER_CLIENT, np.int32)))
        order = np.random.permutation(len(indices))
        clients.append((x_train[indices][order], observed[order]))
    if p_cursor != total_u_positive or n_cursor != total_u_negative:
        raise RuntimeError("controlled allocation does not preserve global totals")
    proxy = (x_train[proxy_idx], y_train[proxy_idx])
    return clients, (x_test, y_test), proxy


def to_device(pair, target_device):
    x, y = pair
    return (torch.from_numpy(np.asarray(x, np.float32)).to(target_device),
            torch.from_numpy(np.asarray(y, np.int32)).to(target_device))


def print_round_result(round_index, acc, recall, auc, f1, threshold=None,
                       specificity=None, bacc=None, fp=None, fn=None):
    """Print the complete test metrics returned for every training round."""
    line = (
        f"Round {round_index + 1:03d} | AUC:{auc:.4f} F1:{f1:.4f} "
        f"Acc:{acc:.4f} Recall:{recall:.4f}"
    )
    if specificity is not None and bacc is not None:
        line += (
            f" Spec:{specificity:.4f} BAcc:{bacc:.4f} "
            f"FP:{int(fp)} FN:{int(fn)}"
        )
    if threshold is not None:
        line += f" Thr:{threshold:.4f}"
    print(line, flush=True)


def run_fedeppu(args, clients, priors, test, proxy):
    x_test, y_test = test
    proxy_x, proxy_y = proxy
    anchor = FedEPPU.BetterCifarCNN(input_channels=3).to(FedEPPU.device)
    model, history, _ = FedEPPU.train_fedavg_proposed(
        client_data=clients, x_test=x_test, t_test=y_test,
        proxy_x=proxy_x, proxy_t=proxy_y, priors=priors,
        anchor_model=anchor, rounds=args.rounds, n_clients=3,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        local_epochs=args.local_epochs, use_amp=True,
        round_callback=print_round_result,
    )
    final_diag = history["client_diag"][-1]
    selected = history["selected"]
    metrics = {"bacc": selected["bacc"], "auc": selected["auc"],
               "threshold": selected["threshold"],
               "selected_round": selected["round"],
               "support": [d["support_aggregation_factor"] for d in final_diag],
               "base_weight": [d["base_aggregation_weight"] for d in final_diag],
               "effective_weight": [d["effective_aggregation_weight"] for d in final_diag]}
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
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
    target_device = FedEPPU.device
    clients = [to_device(pair, target_device) for pair in raw_clients]
    test = to_device(raw_test, target_device)
    proxy = to_device(raw_proxy, target_device)
    theoretical_support = [min(1.0, (1.0 - p) * 192) for p in priors]
    print(f"method=fedeppu; config={args.prior_config}; priors={priors}; seed={args.seed}")
    print(f"client_sizes={[len(x) for x, _ in clients]}; P={P_PER_CLIENT}; U={U_PER_CLIENT}")
    print(f"global_U_positive={int(sum(priors)*U_PER_CLIENT)}; global_U_negative={3*U_PER_CLIENT-int(sum(priors)*U_PER_CLIENT)}")
    print(f"theoretical_batch_support={theoretical_support}")
    model, metrics = run_fedeppu(args, clients, priors, test, proxy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": "fedeppu", "prior_config": args.prior_config,
               "priors": list(priors), "seed": args.seed,
               "p_per_client": P_PER_CLIENT, "u_per_client": U_PER_CLIENT,
               "theoretical_support": theoretical_support, "metrics": metrics}
    torch.save({"model": model.state_dict(), **payload}, output)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
