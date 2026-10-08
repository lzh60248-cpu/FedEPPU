"""Section 4.4.3: summarize and plot the paired PU-ranking ablation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("full", "no_ranking")
SEEDS = (42, 43, 44, 45, 46)
LABELS = {
    "full": "Full FedEPPU",
    "no_ranking": "FedEPPU w/o PU ranking",
}
COLORS = {"full": "#111111", "no_ranking": "#0072B2"}


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
        raise FileNotFoundError(f"Missing PU-ranking result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload["history"]
    if len(history["auc"]) != 200 or len(history["bacc"]) != 200:
        raise ValueError(f"Expected 200 rounds in {path}")
    selected = history["selected"]
    auc = np.asarray(history["auc"], dtype=float) * 100.0
    bacc = np.asarray(history["bacc"], dtype=float) * 100.0
    result = {
        metric: float(selected[metric]) * 100.0
        for metric in ("acc", "f1", "bacc", "auc")
    }
    result.update({
        "max_auc": float(auc.max()),
        "last20_auc": float(auc[-20:].mean()),
        "auc_last20_sd": sample_sd(auc[-20:]),
        "auc_last20_range": float(np.ptp(auc[-20:])),
        "bacc_last20_sd": sample_sd(bacc[-20:]),
        "bacc_last20_range": float(np.ptp(bacc[-20:])),
        "auc_curve": auc,
        "bacc_curve": bacc,
    })
    return result


def mean_sd(runs, key):
    values = [run[key] for run in runs]
    return float(np.mean(values)), sample_sd(values)


def fmt(pair):
    return f"{pair[0]:.2f} ± {pair[1]:.2f}"


def write_tables(output, runs):
    performance_keys = ("acc", "f1", "bacc", "auc", "max_auc", "last20_auc")
    stability_keys = (
        "auc_last20_sd", "auc_last20_range",
        "bacc_last20_sd", "bacc_last20_range",
    )
    summary = {
        variant: {
            key: mean_sd(runs[variant], key)
            for key in performance_keys + stability_keys
        }
        for variant in VARIANTS
    }
    csv_rows = []
    for variant in VARIANTS:
        row = {"variant": variant, "label": LABELS[variant]}
        for key, (mean, sd) in summary[variant].items():
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
        csv_rows.append(row)
    with (output / "ranking_ablation_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    performance = [
        "| Variant | ACC (%) | F1 (%) | BAcc (%) | AUC (%) | Max AUC (%) | Last-20 AUC (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        performance.append("| " + " | ".join(
            [LABELS[variant]] + [fmt(summary[variant][key]) for key in performance_keys]
        ) + " |")
    contributions = [
        summary["full"][key][0] - summary["no_ranking"][key][0]
        for key in performance_keys
    ]
    performance.append(
        "| PU-ranking contribution | " +
        " | ".join(f"{value:+.2f}" for value in contributions) + " |"
    )
    (output / "ranking_performance_contribution_table.md").write_text(
        "\n".join(performance) + "\n", encoding="utf-8"
    )

    stability = [
        "| Variant | AUC last-20 SD | AUC last-20 range | BAcc last-20 SD | BAcc last-20 range |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        stability.append("| " + " | ".join(
            [LABELS[variant]] + [fmt(summary[variant][key]) for key in stability_keys]
        ) + " |")
    stability_gain = [
        summary["no_ranking"][key][0] - summary["full"][key][0]
        for key in stability_keys
    ]
    stability.append(
        "| PU-ranking stability contribution | " +
        " | ".join(f"{value:+.2f}" for value in stability_gain) + " |"
    )
    (output / "ranking_stability_contribution_table.md").write_text(
        "\n".join(stability) + "\n", encoding="utf-8"
    )


def plot_curves(output, runs):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#B8B8B8",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True)
    rounds = np.arange(1, 201)
    for axis, key, title in zip(
            axes, ("auc_curve", "bacc_curve"), ("AUC", "Balanced accuracy")):
        for variant in VARIANTS:
            matrix = np.stack([run[key] for run in runs[variant]])
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1)
            axis.plot(
                rounds, mean, color=COLORS[variant],
                linestyle="-" if variant == "full" else "--",
                linewidth=2.5, label=LABELS[variant],
            )
            axis.fill_between(
                rounds, mean - sd, mean + sd,
                color=COLORS[variant], alpha=.14,
            )
        axis.set_title(title, fontweight="semibold")
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Metric (%)")
        axis.set_xlim(1, 200)
        axis.grid(True, color="#D9D9D9", linewidth=.6, alpha=.75)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.03))
    fig.suptitle("Contribution of PU ranking on CIFAR-10", fontsize=15,
                 fontweight="bold", y=1.09)
    fig.tight_layout(rect=(0, 0, 1, .91))
    png = output / "ranking_ablation_round_curves.png"
    pdf = output / "ranking_ablation_round_curves.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figure={png}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        variant: [load_run(args.input_dir, variant, seed) for seed in SEEDS]
        for variant in VARIANTS
    }
    write_tables(args.output_dir, runs)
    plot_curves(args.output_dir, runs)
    print("validated_results=10")
    print(f"artifacts={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
