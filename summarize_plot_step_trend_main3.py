"""Summarize and plot the CIFAR-10 Main3 local-step trend experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_CAPS = (5, 10, 15, 20, 30)
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("bacc", "auc", "f1")
COLORS = {"bacc": "#0072B2", "auc": "#D55E00", "f1": "#009E73"}
LABELS = {"bacc": "BAcc", "auc": "AUC", "f1": "F1"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def actual_steps(history):
    """Return each client's mean optimizer steps per local epoch over all rounds."""
    diagnostics = history.get("client_diag", [])
    if not diagnostics:
        raise ValueError("client_diag is missing; actual step usage cannot be verified")
    n_clients = len(diagnostics[0])
    per_client = []
    for client in range(n_clients):
        values = [float(round_diag[client]["actual_steps_per_epoch"])
                  for round_diag in diagnostics]
        per_client.append(float(np.mean(values)))
    return per_client


def main():
    args = parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for step_cap in STEP_CAPS:
        for seed in SEEDS:
            folder = args.input_dir / f"steps{step_cap}" / f"seed{seed}"
            files = sorted(folder.glob("*.json"))
            if len(files) != 1:
                missing.append(f"steps{step_cap}/seed{seed}: found {len(files)} JSON")
                continue
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            history = payload["history"]
            selected = history["selected"]
            client_steps = actual_steps(history)
            row = {
                "step_cap": step_cap,
                "seed": seed,
                "selected_round": int(selected["round"]),
                "client1_actual_steps_per_epoch": client_steps[0],
                "client2_actual_steps_per_epoch": client_steps[1],
                "client3_actual_steps_per_epoch": client_steps[2],
                "mean_actual_steps_per_epoch": float(np.mean(client_steps)),
            }
            for metric in METRICS:
                curve = np.asarray(history[metric], dtype=float) * 100.0
                row[f"selected_{metric}"] = float(selected[metric]) * 100.0
                row[f"last_round_{metric}"] = float(curve[-1])
                row[f"peak_{metric}"] = float(curve.max())
                row[f"last20_{metric}_mean"] = float(curve[-20:].mean())
                row[f"last20_{metric}_sd"] = float(curve[-20:].std(ddof=1))
                row[f"peak_to_last20_{metric}_drop"] = float(
                    curve.max() - curve[-20:].mean()
                )
                row[f"peak_to_last_round_{metric}_drop"] = float(
                    curve.max() - curve[-1]
                )
            records.append(row)
    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))

    fields = list(records[0].keys())
    per_seed_path = output / "step_trend_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)

    summaries = []
    for step_cap in STEP_CAPS:
        group = [row for row in records if row["step_cap"] == step_cap]
        summary = {"step_cap": step_cap}
        for field in ("mean_actual_steps_per_epoch", *[f"selected_{m}" for m in METRICS],
                      *[f"last20_{m}_sd" for m in METRICS],
                      *[f"peak_to_last20_{m}_drop" for m in METRICS],
                      *[f"peak_to_last_round_{m}_drop" for m in METRICS]):
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(
                [row[field] for row in group]
            )
        summaries.append(summary)

    summary_path = output / "step_trend_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)

    table = [
        "| Step cap | Actual steps/epoch | BAcc (%) | AUC (%) | F1 (%) | Last-20 AUC SD | Peak-to-final AUC drop |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cells = [
            str(row["step_cap"]),
            f'{row["mean_actual_steps_per_epoch_mean"]:.2f} ± {row["mean_actual_steps_per_epoch_sd"]:.2f}',
            f'{row["selected_bacc_mean"]:.2f} ± {row["selected_bacc_sd"]:.2f}',
            f'{row["selected_auc_mean"]:.2f} ± {row["selected_auc_sd"]:.2f}',
            f'{row["selected_f1_mean"]:.2f} ± {row["selected_f1_sd"]:.2f}',
            f'{row["last20_auc_sd_mean"]:.3f}',
            f'{row["peak_to_last_round_auc_drop_mean"]:.2f}',
        ]
        table.append("| " + " | ".join(cells) + " |")
    table_path = output / "step_trend_table.md"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")

    x = np.asarray(STEP_CAPS, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for metric in METRICS:
        y = [row[f"selected_{metric}_mean"] for row in summaries]
        e = [row[f"selected_{metric}_sd"] for row in summaries]
        axes[0].errorbar(x, y, yerr=e, marker="o", linewidth=2.0, capsize=3,
                         color=COLORS[metric], label=LABELS[metric])
    axes[0].set_ylabel("Selected-checkpoint performance (%)")
    axes[0].legend(frameon=False)

    axes[1].plot(x, [row["last20_auc_sd_mean"] for row in summaries],
                 marker="o", linewidth=2.2, color=COLORS["auc"])
    axes[1].set_ylabel("Mean late-round AUC SD (percentage points)")

    axes[2].plot(x, [row["peak_to_last_round_auc_drop_mean"] for row in summaries],
                 marker="o", linewidth=2.2, color="#CC79A7")
    axes[2].set_ylabel("Peak-to-final AUC drop (points)")

    for axis, title in zip(axes, ("Performance", "Late-round stability", "Late-round degradation")):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Maximum local steps per epoch")
        axis.set_xticks(STEP_CAPS)
        axis.grid(True, alpha=0.25)
    fig.savefig(output / "step_trend.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "step_trend.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Per-seed results -> {per_seed_path}")
    print(f"Summary -> {summary_path}")
    print(f"Paper table -> {table_path}")
    print(f"Figure -> {output / 'step_trend.png'}")


if __name__ == "__main__":
    main()
