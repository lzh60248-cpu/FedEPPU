"""Summarize and plot Main3 component performance and stability contributions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("full", "no_ranking", "no_complement", "no_support", "fedavg_aggregation")
SEEDS = (42, 43, 44, 45, 46)
LABELS = {
    "full": "Full FedEPPU",
    "no_ranking": "w/o PU ranking",
    "no_complement": "w/o complementary mass",
    "no_support": "w/o negative support",
    "fedavg_aggregation": "FedAvg aggregation",
}
COLORS = {
    "full": "#111111",
    "no_ranking": "#0072B2",
    "no_complement": "#E69F00",
    "no_support": "#009E73",
    "fedavg_aggregation": "#CC79A7",
}
PERFORMANCE = ("acc", "f1", "bacc", "auc")
STABILITY = ("bacc_sd", "bacc_range", "auc_sd", "auc_range")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def sample_sd(values):
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def load_run(root, variant, seed):
    path = root / variant / f"seed{seed}" / f"{variant}_cifar10_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ablation result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload["history"]
    if len(history["auc"]) != 200:
        raise ValueError(f"Expected 200 rounds in {path}")
    selected = history["selected"]
    result = {key: float(selected[key]) * 100.0 for key in PERFORMANCE}
    auc = np.asarray(history["auc"], dtype=float) * 100.0
    bacc = np.asarray(history["bacc"], dtype=float) * 100.0
    result.update({
        "max_auc": float(auc.max()),
        "last20_auc": float(auc[-20:].mean()),
        "bacc_sd": sample_sd(bacc[-20:]),
        "bacc_range": float(np.ptp(bacc[-20:])),
        "auc_sd": sample_sd(auc[-20:]),
        "auc_range": float(np.ptp(auc[-20:])),
        "auc_curve": auc,
        "bacc_curve": bacc,
    })
    return result


def fmt(mean, sd):
    return f"{mean:.2f} ± {sd:.2f}"


def save_tables(output, runs):
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for metric in PERFORMANCE + ("max_auc", "last20_auc") + STABILITY:
            values = [run[metric] for run in runs[variant]]
            summary[variant][metric] = (
                float(np.mean(values)), sample_sd(values)
            )

    csv_rows = []
    for variant in VARIANTS:
        row = {"variant": variant, "label": LABELS[variant]}
        for metric, (mean, sd) in summary[variant].items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        csv_rows.append(row)
    with (output / "component_ablation_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    performance_table = [
        "| Variant | ACC (%) | F1 (%) | BAcc (%) | AUC (%) | Max AUC (%) | Last-20 AUC (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        cells = [LABELS[variant]] + [
            fmt(*summary[variant][metric])
            for metric in PERFORMANCE + ("max_auc", "last20_auc")
        ]
        performance_table.append("| " + " | ".join(cells) + " |")
    (output / "component_performance_table.md").write_text(
        "\n".join(performance_table) + "\n", encoding="utf-8"
    )

    contribution_table = [
        "| Component | ACC contribution | F1 contribution | BAcc contribution | AUC contribution |",
        "|---|---:|---:|---:|---:|",
    ]
    component_variants = (
        ("PU ranking", "no_ranking"),
        ("Complementary mass", "no_complement"),
        ("Negative support", "no_support"),
        ("Joint aggregation design", "fedavg_aggregation"),
    )
    for component, ablation in component_variants:
        values = [
            summary["full"][metric][0] - summary[ablation][metric][0]
            for metric in PERFORMANCE
        ]
        contribution_table.append(
            "| " + component + " | " + " | ".join(f"{value:+.2f}" for value in values) + " |"
        )
    (output / "component_numerical_contribution_table.md").write_text(
        "\n".join(contribution_table) + "\n", encoding="utf-8"
    )

    stability_table = [
        "| Variant | BAcc last-20 SD | BAcc last-20 range | AUC last-20 SD | AUC last-20 range |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        cells = [LABELS[variant]] + [fmt(*summary[variant][metric]) for metric in STABILITY]
        stability_table.append("| " + " | ".join(cells) + " |")
    (output / "component_stability_table.md").write_text(
        "\n".join(stability_table) + "\n", encoding="utf-8"
    )

    stability_contribution = [
        "| Component | BAcc SD contribution | BAcc range contribution | AUC SD contribution | AUC range contribution |",
        "|---|---:|---:|---:|---:|",
    ]
    for component, ablation in component_variants:
        values = [
            summary[ablation][metric][0] - summary["full"][metric][0]
            for metric in STABILITY
        ]
        stability_contribution.append(
            "| " + component + " | " + " | ".join(f"{value:+.2f}" for value in values) + " |"
        )
    (output / "component_stability_contribution_table.md").write_text(
        "\n".join(stability_contribution) + "\n", encoding="utf-8"
    )
    return summary


def plot_bars(output, summary, metrics, titles, filename, ylabel):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    x = np.arange(len(VARIANTS))
    for axis, metric, title in zip(axes.flat, metrics, titles):
        means = [summary[v][metric][0] for v in VARIANTS]
        errors = [summary[v][metric][1] for v in VARIANTS]
        axis.bar(x, means, yerr=errors, capsize=4,
                 color=[COLORS[v] for v in VARIANTS], alpha=.88,
                 edgecolor="white", linewidth=.7)
        axis.set_title(title, fontweight="semibold")
        axis.set_xticks(x, [LABELS[v] for v in VARIANTS], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=.6, alpha=.75)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_curves(output, runs):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), sharex=True)
    rounds = np.arange(1, 201)
    for axis, metric, title in zip(
            axes, ("bacc_curve", "auc_curve"), ("Balanced accuracy", "AUC")):
        for variant in VARIANTS:
            matrix = np.stack([run[metric] for run in runs[variant]])
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1)
            axis.plot(rounds, mean, color=COLORS[variant],
                      linewidth=2.5 if variant == "full" else 1.6,
                      label=LABELS[variant])
            axis.fill_between(rounds, mean - sd, mean + sd,
                              color=COLORS[variant], alpha=.09)
        axis.set_title(title, fontweight="semibold")
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Metric (%)")
        axis.set_xlim(1, 200)
        axis.grid(True, color="#D9D9D9", linewidth=.6, alpha=.75)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output / "component_ablation_round_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "component_ablation_round_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        variant: [load_run(args.input_dir, variant, seed) for seed in SEEDS]
        for variant in VARIANTS
    }
    summary = save_tables(args.output_dir, runs)
    plot_bars(
        args.output_dir, summary, PERFORMANCE,
        ("Accuracy", "F1 score", "Balanced accuracy", "AUC"),
        "component_ablation_performance", "Performance (%)",
    )
    plot_bars(
        args.output_dir, summary, STABILITY,
        ("BAcc last-20 SD", "BAcc last-20 range", "AUC last-20 SD", "AUC last-20 range"),
        "component_ablation_stability", "Instability (percentage points)",
    )
    plot_curves(args.output_dir, runs)
    print("validated_results=25")
    print(f"artifacts={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
