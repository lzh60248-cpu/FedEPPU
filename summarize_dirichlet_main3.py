"""Summarize the 15-run CIFAR-10 FedEPPU Dirichlet experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("cifar10",)
ALPHAS = (("alpha0p1", 0.1), ("alpha0p5", 0.5), ("alpha1p0", 1.0))
LABELS = {
    "cifar10": "CIFAR-10",
}
METRICS = ("bacc", "acc", "auc", "f1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    args = parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []
    for dataset in DATASETS:
        for alpha_name, alpha in ALPHAS:
            for seed in range(42, 47):
                folder = args.input_dir / dataset / alpha_name / f"seed{seed}"
                files = sorted(folder.glob("*.json"))
                if len(files) != 1:
                    missing.append(f"{dataset}/{alpha_name}/seed{seed}: found {len(files)} JSON")
                    continue
                payload = json.loads(files[0].read_text(encoding="utf-8"))
                history = payload["history"]
                selected = history["selected"]
                row = {"dataset": dataset, "alpha": alpha, "seed": seed}
                for metric in METRICS:
                    row[metric] = float(selected[metric]) * 100.0
                    late = np.asarray(history[metric][-20:], dtype=float) * 100.0
                    row[f"last20_{metric}_mean"] = float(late.mean())
                    row[f"last20_{metric}_sd"] = float(late.std(ddof=1))
                row["selected_round"] = int(selected["round"])
                row["client_priors"] = payload["metadata"]["client_priors"]
                row["client_sample_counts"] = payload["metadata"]["client_sample_counts"]
                records.append(row)
    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))

    per_seed = output / "dirichlet_per_seed.csv"
    fields = ["dataset", "alpha", "seed", *METRICS, "selected_round",
              "last20_bacc_mean", "last20_bacc_sd", "last20_acc_mean", "last20_acc_sd",
              "last20_auc_mean", "last20_auc_sd", "last20_f1_mean", "last20_f1_sd",
              "client_priors", "client_sample_counts"]
    with per_seed.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records:
            saved = dict(row)
            saved["client_priors"] = json.dumps(saved["client_priors"])
            saved["client_sample_counts"] = json.dumps(saved["client_sample_counts"])
            writer.writerow(saved)

    summary_rows = []
    for dataset in DATASETS:
        for _, alpha in ALPHAS:
            group = [r for r in records if r["dataset"] == dataset and r["alpha"] == alpha]
            summary = {"dataset": LABELS[dataset], "alpha": alpha}
            for metric in METRICS:
                summary[f"{metric}_mean"], summary[f"{metric}_sd"] = mean_sd(
                    [r[metric] for r in group]
                )
            summary["last20_auc_temporal_sd_mean"] = float(np.mean(
                [r["last20_auc_sd"] for r in group]
            ))
            summary["last20_bacc_temporal_sd_mean"] = float(np.mean(
                [r["last20_bacc_sd"] for r in group]
            ))
            summary_rows.append(summary)

    summary_csv = output / "dirichlet_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)

    table = [
        "| Dataset | Dirichlet α | BAcc (%) | ACC (%) | AUC (%) | F1 (%) | Last-20 AUC SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        cells = [row["dataset"], f'{row["alpha"]:g}']
        for metric in METRICS:
            cells.append(f'{row[f"{metric}_mean"]:.2f} ± {row[f"{metric}_sd"]:.2f}')
        cells.append(f'{row["last20_auc_temporal_sd_mean"]:.3f}')
        table.append("| " + " | ".join(cells) + " |")
    table_path = output / "dirichlet_table.md"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    colors = ("#0072B2",)
    for dataset, color in zip(DATASETS, colors):
        rows = [r for r in summary_rows if r["dataset"] == LABELS[dataset]]
        x = [r["alpha"] for r in rows]
        for axis, metric, title in zip(axes, ("bacc", "auc"), ("BAcc", "AUC")):
            y = [r[f"{metric}_mean"] for r in rows]
            e = [r[f"{metric}_sd"] for r in rows]
            axis.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3,
                          color=color, label=LABELS[dataset])
            axis.set_xscale("log")
            axis.set_xticks([0.1, 0.5, 1.0], labels=["0.1", "0.5", "1.0"])
            axis.set_xlabel("Dirichlet concentration α")
            axis.set_ylabel(f"{title} (%)")
            axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "dirichlet_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "dirichlet_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Per-seed CSV -> {per_seed}")
    print(f"Summary CSV -> {summary_csv}")
    print(f"Paper table -> {table_path}")
    print(f"Figure -> {output / 'dirichlet_performance.png'}")


if __name__ == "__main__":
    main()
