"""Build Table 4 and Figure 10 for the CIFAR-10 Dirichlet comparison."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = (("alpha0p1", 0.1), ("alpha0p5", 0.5), ("alpha1p0", 1.0), ("alpha2p0", 2.0))
METHODS = ("fedavg_nnpu", "fedeppu")
METHOD_LABELS = {"fedavg_nnpu": "FedAvg-nnPU", "fedeppu": "FedEPPU"}
SEEDS = (42, 43, 44, 45, 46)
COLORS = {"fedavg_nnpu": "#7F7F7F", "fedeppu": "#0072B2"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def load_one(root, alpha_name, method, seed):
    files = list((root / alpha_name / method / f"seed{seed}").glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one JSON for {alpha_name}/{method}/seed{seed}, found {len(files)}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    if method == "fedeppu":
        history = data["history"]
        selected = history["selected"]
        metrics = {key: float(selected[key]) for key in ("bacc", "auc")}
        auc = np.asarray(history["auc"], dtype=float)
        priors = np.asarray(data["metadata"]["client_priors"], dtype=float)
        counts = np.asarray(data["metadata"]["client_sample_counts"], dtype=int)
    else:
        history = data["history"]
        metrics = {key: float(data["metrics"][key]) for key in ("bacc", "auc")}
        auc = np.asarray([row["auc"] for row in history], dtype=float)
        priors = np.asarray(data["client_priors"], dtype=float)
        counts = np.asarray(data["client_sample_counts"], dtype=int)
    if len(auc) != 200:
        raise ValueError(f"Expected 200 rounds, found {len(auc)}")
    return {
        "bacc": metrics["bacc"], "auc": metrics["auc"],
        "last20_auc": float(auc[-20:].mean()),
        "last20_auc_sd": float(auc[-20:].std(ddof=1)),
        "peak_drop": float(auc.max() - auc[-20:].mean()),
        "priors": priors, "counts": counts,
    }


def display(values):
    values = np.asarray(values, dtype=float) * 100.0
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def main():
    args = parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    grouped, missing = {}, []
    for alpha_name, alpha in CONFIGS:
        for method in METHODS:
            values = []
            for seed in SEEDS:
                try:
                    values.append(load_one(args.input_dir, alpha_name, method, seed))
                except (FileNotFoundError, ValueError) as error:
                    missing.append(str(error))
            if len(values) == len(SEEDS):
                grouped[(alpha, method)] = values
    if missing:
        raise SystemExit("Incomplete Dirichlet comparison:\n" + "\n".join(missing))

    rows = []
    for _, alpha in CONFIGS:
        for method in METHODS:
            values = grouped[(alpha, method)]
            priors = np.concatenate([item["priors"] for item in values])
            counts = np.concatenate([item["counts"] for item in values])
            rows.append({
                "alpha": alpha,
                "method": METHOD_LABELS[method],
                "actual_prior_range": f"[{priors.min():.4f}, {priors.max():.4f}]",
                "actual_sample_range": f"[{counts.min()}, {counts.max()}]",
                "bacc": display([item["bacc"] for item in values]),
                "auc": display([item["auc"] for item in values]),
                "last20_auc": display([item["last20_auc"] for item in values]),
                "last20_auc_sd": f"{np.mean([item['last20_auc_sd'] for item in values]) * 100.0:.3f}",
                "peak_drop": f"{np.mean([item['peak_drop'] for item in values]) * 100.0:.2f}",
            })

    with (output / "dirichlet_comparison_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    table = [
        "| Dirichlet α | Method | Actual prior range | Actual sample range | BAcc (%) | AUC (%) | Last-20 AUC (%) | Last-20 AUC SD | Peak drop |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append("| " + " | ".join([
            f'{row["alpha"]:g}', row["method"], row["actual_prior_range"], row["actual_sample_range"],
            row["bacc"], row["auc"], row["last20_auc"], row["last20_auc_sd"], row["peak_drop"],
        ]) + " |")
    (output / "dirichlet_comparison_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    alphas = [alpha for _, alpha in CONFIGS]
    for method in METHODS:
        values = [grouped[(alpha, method)] for alpha in alphas]
        auc_mean = [np.mean([item["auc"] for item in group]) * 100.0 for group in values]
        auc_sd = [np.std([item["auc"] for item in group], ddof=1) * 100.0 for group in values]
        late_sd = [np.mean([item["last20_auc_sd"] for item in group]) * 100.0 for group in values]
        axes[0].errorbar(alphas, auc_mean, yerr=auc_sd, marker="o", capsize=3, linewidth=2,
                         color=COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(alphas, late_sd, marker="o", linewidth=2,
                     color=COLORS[method], label=METHOD_LABELS[method])
    for axis, title, ylabel in zip(
        axes,
        ("Final AUC", "Last-20-round AUC stability"),
        ("AUC (%)", "Mean temporal AUC SD (points)"),
    ):
        axis.set_xscale("log")
        axis.set_xticks(alphas, labels=[f"{alpha:g}" for alpha in alphas])
        axis.set_xlabel("Dirichlet concentration α")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False)
    figure.savefig(output / "dirichlet_comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "dirichlet_comparison.pdf", bbox_inches="tight")
    print(f"table={output / 'dirichlet_comparison_table.md'}")
    print(f"figure={output / 'dirichlet_comparison.png'}")


if __name__ == "__main__":
    main()
