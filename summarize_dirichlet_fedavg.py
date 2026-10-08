"""Summarize four-concentration FedAvg-nnPU Dirichlet results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = (("alpha0p1", 0.1), ("alpha0p5", 0.5), ("alpha1p0", 1.0), ("alpha2p0", 2.0))
SEEDS = (42, 43, 44, 45, 46)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def display(values):
    values = np.asarray(values, dtype=float) * 100.0
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def main():
    args = parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for name, alpha in CONFIGS:
        for seed in SEEDS:
            file = args.input_dir / name / f"seed{seed}" / f"fedavg_nnpu_cifar10_{name}_seed{seed}.json"
            if not file.is_file():
                missing.append(str(file))
                continue
            payload = json.loads(file.read_text(encoding="utf-8"))
            records.append(payload)
    if missing:
        raise SystemExit("Missing results:\n" + "\n".join(missing))

    summary = []
    for name, alpha in CONFIGS:
        group = [item for item in records if item["alpha"] == alpha]
        priors = np.concatenate([item["client_priors"] for item in group])
        counts = np.concatenate([item["client_sample_counts"] for item in group])
        row = {
            "alpha": alpha,
            "actual_prior_range": f"[{priors.min():.4f}, {priors.max():.4f}]",
            "actual_sample_range": f"[{counts.min()}, {counts.max()}]",
            "bacc": display([item["metrics"]["bacc"] for item in group]),
            "auc": display([item["metrics"]["auc"] for item in group]),
            "last20_auc": display([item["last20_auc_mean"] for item in group]),
            "last20_auc_sd": f"{np.mean([item['last20_auc_sd'] for item in group]) * 100.0:.3f}",
            "peak_drop": f"{np.mean([item['peak_to_last20_auc_drop'] for item in group]) * 100.0:.2f}",
        }
        summary.append(row)

    with (output / "dirichlet_fedavg_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    table = [
        "| Dirichlet α | Method | Actual prior range | Actual sample range | BAcc (%) | AUC (%) | Last-20 AUC (%) | Last-20 AUC SD | Peak drop |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        table.append("| " + " | ".join([
            f'{row["alpha"]:g}', "FedAvg-nnPU", row["actual_prior_range"], row["actual_sample_range"],
            row["bacc"], row["auc"], row["last20_auc"], row["last20_auc_sd"], row["peak_drop"],
        ]) + " |")
    (output / "dirichlet_fedavg_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    x = [row["alpha"] for row in summary]
    auc = [float(row["auc"].split()[0]) for row in summary]
    stability = [float(row["last20_auc_sd"]) for row in summary]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    axes[0].plot(x, auc, marker="o", linewidth=2, color="#0072B2")
    axes[0].set_ylabel("Final AUC (%)")
    axes[1].plot(x, stability, marker="o", linewidth=2, color="#D55E00")
    axes[1].set_ylabel("Mean last-20 AUC SD (points)")
    for axis, title in zip(axes, ("FedAvg-nnPU performance", "FedAvg-nnPU stability")):
        axis.set_xscale("log")
        axis.set_xticks(x, labels=[f"{value:g}" for value in x])
        axis.set_xlabel("Dirichlet concentration α")
        axis.set_title(title, fontweight="bold")
        axis.grid(alpha=.25)
    figure.savefig(output / "dirichlet_fedavg.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "dirichlet_fedavg.pdf", bbox_inches="tight")
    print(f"table={output / 'dirichlet_fedavg_table.md'}")
    print(f"figure={output / 'dirichlet_fedavg.png'}")


if __name__ == "__main__":
    main()
