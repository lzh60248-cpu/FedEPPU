"""Summarize Section 4.5 synchronized per-round wall-clock measurements."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")
METHODS = ("fedavg_nnpu", "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu")
LABELS = {
    "fashion_mnist": "Fashion-MNIST (benchmark)",
    "cifar10": "CIFAR-10 (benchmark)",
    "chest": "Chest X-ray (real medical)",
    "kolektor_sdd2": "KolektorSDD2 (real industrial)",
    "fedavg_nnpu": "FedAvg-nnPU", "fedprox_nnpu": "FedProx-nnPU",
    "scaffold_nnpu": "SCAFFOLD-nnPU", "fedpu": "FedPU",
    "mrfl_nnpu": "MRFL-nnPU", "fedeppu": "FedEPPU",
}


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--warmup_rounds", type=int, default=10)
    return p.parse_args()


def load_one(path, warmup):
    data = json.loads(path.read_text(encoding="utf-8"))
    timing = data.get("timing") or data.get("history", {}).get("timing")
    if not timing or len(timing) <= warmup:
        raise ValueError(f"insufficient timing records: {path}")
    values = np.asarray([x["train_aggregate_seconds"] for x in timing[warmup:]], float)
    metadata = data.get("metadata", data)
    params = int(metadata["model_parameters"])
    equivalents = int(metadata["communication_model_equivalents_per_round"])
    return float(values.mean()), params * 4 * equivalents / (1024 ** 2)


def fmt(values):
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def main():
    args = args_parser()
    source = Path(args.input_dir)
    output = Path(args.output_dir or source / "summary")
    output.mkdir(parents=True, exist_ok=True)
    stats, payload = {}, {}
    missing = []
    for dataset in DATASETS:
        for method in METHODS:
            values, sizes = [], []
            for seed in range(42, 47):
                files = sorted((source / dataset / method / f"seed{seed}").glob("*.json"))
                if len(files) != 1:
                    missing.append(f"{dataset}/{method}/seed{seed}: found {len(files)} JSON")
                    continue
                value, mb = load_one(files[0], args.warmup_rounds)
                values.append(value); sizes.append(mb)
            if len(values) == 5:
                stats[(dataset, method)] = values
                payload[(dataset, method)] = sizes
    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))

    csv_path = output / "communication_time_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset"] + [LABELS[m] for m in METHODS] + ["FedEPPU vs FedAvg"])
        for dataset in DATASETS:
            base = np.mean(stats[(dataset, "fedavg_nnpu")])
            ours = np.mean(stats[(dataset, "fedeppu")])
            writer.writerow([LABELS[dataset]] + [fmt(stats[(dataset, m)]) for m in METHODS]
                            + [f"{(ours / base - 1) * 100:+.1f}%"])
        averages = [np.mean([np.mean(stats[(d, m)]) for d in DATASETS]) for m in METHODS]
        writer.writerow(["Average"] + [f"{v:.3f}" for v in averages]
                        + [f"{(averages[-1] / averages[0] - 1) * 100:+.1f}%"])

    md = ["| Dataset | " + " | ".join(LABELS[m] for m in METHODS) + " | FedEPPU vs FedAvg |",
          "|---|" + "---:|" * (len(METHODS) + 1)]
    for dataset in DATASETS:
        base = np.mean(stats[(dataset, "fedavg_nnpu")]); ours = np.mean(stats[(dataset, "fedeppu")])
        md.append("| " + LABELS[dataset] + " | " + " | ".join(fmt(stats[(dataset, m)]) for m in METHODS)
                  + f" | {(ours / base - 1) * 100:+.1f}% |")
    averages = [np.mean([np.mean(stats[(d, m)]) for d in DATASETS]) for m in METHODS]
    md.append("| Average | " + " | ".join(f"{v:.3f}" for v in averages)
              + f" | {(averages[-1] / averages[0] - 1) * 100:+.1f}% |")
    (output / "communication_time_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (output / "communication_payload_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["Dataset"] + [LABELS[m] for m in METHODS])
        for dataset in DATASETS:
            writer.writerow([LABELS[dataset]] + [f"{np.mean(payload[(dataset, m)]):.2f}" for m in METHODS])
    print(f"Seeds: 42-46; warm-up rounds excluded: 1-{args.warmup_rounds}; measured: {args.warmup_rounds+1}-200")
    print(f"Time table (seconds/round, mean ± sample SD) -> {csv_path}")
    print(f"Markdown -> {output / 'communication_time_table.md'}")
    print(f"Theoretical payload (MiB/round) -> {output / 'communication_payload_table.csv'}")


if __name__ == "__main__":
    main()
