"""Summarize Main3 positive-allocation-ratio sensitivity runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RHO_SPECS = (("0125", 0.125), ("0250", 0.25), ("0375", 0.375), ("0500", 0.5))
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("acc", "f1", "bacc", "auc")


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for rho_name, rho in RHO_SPECS:
        for seed in SEEDS:
            folder = args.input_dir / f"rho{rho_name}" / f"seed{seed}"
            files = sorted(folder.glob("*.json"))
            if len(files) != 1:
                missing.append(f"rho{rho_name}/seed{seed}: found {len(files)} JSON files")
                continue
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            history = payload["history"]
            selected = history["selected"]
            diagnostics = history.get("client_diag", [])
            observed_p = sorted({
                int(client["positive_per_batch"])
                for round_diag in diagnostics for client in round_diag
            })
            observed_u = sorted({
                int(client["unlabeled_per_batch"])
                for round_diag in diagnostics for client in round_diag
            })
            if len(observed_p) != 1 or len(observed_u) != 1:
                raise ValueError(f"Inconsistent batch composition in {files[0]}")
            row = {
                "rho": rho, "seed": seed,
                "positive_per_batch": observed_p[0],
                "unlabeled_per_batch": observed_u[0],
                "selected_round": int(selected["round"]),
            }
            for metric in METRICS:
                curve = np.asarray(history[metric], dtype=float) * 100.0
                row[f"selected_{metric}"] = float(selected[metric]) * 100.0
                row[f"last20_{metric}_mean"] = float(curve[-20:].mean())
                row[f"last20_{metric}_sd"] = float(curve[-20:].std(ddof=1))
                row[f"peak_to_final_{metric}_drop"] = float(curve.max() - curve[-1])
            rows.append(row)
    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))

    per_seed = output / "rho_sensitivity_per_seed.csv"
    with per_seed.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    summaries = []
    for _, rho in RHO_SPECS:
        group = [row for row in rows if row["rho"] == rho]
        summary = {
            "rho": rho,
            "positive_per_batch": group[0]["positive_per_batch"],
            "unlabeled_per_batch": group[0]["unlabeled_per_batch"],
        }
        for field in (
            *[f"selected_{m}" for m in METRICS],
            "last20_auc_mean", "last20_auc_sd", "peak_to_final_auc_drop",
        ):
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(
                [row[field] for row in group]
            )
        summaries.append(summary)

    summary_csv = output / "rho_sensitivity_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)

    table = [
        "| rho | P/U per batch | ACC (%) | F1 (%) | BAcc (%) | AUC (%) | Last-20 AUC | Late AUC SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            f'| {row["rho"]:.3f} | {row["positive_per_batch"]}/{row["unlabeled_per_batch"]} '
            f'| {row["selected_acc_mean"]:.2f} ± {row["selected_acc_sd"]:.2f} '
            f'| {row["selected_f1_mean"]:.2f} ± {row["selected_f1_sd"]:.2f} '
            f'| {row["selected_bacc_mean"]:.2f} ± {row["selected_bacc_sd"]:.2f} '
            f'| {row["selected_auc_mean"]:.2f} ± {row["selected_auc_sd"]:.2f} '
            f'| {row["last20_auc_mean_mean"]:.2f} ± {row["last20_auc_mean_sd"]:.2f} '
            f'| {row["last20_auc_sd_mean"]:.3f} |'
        )
    table_path = output / "rho_sensitivity_table.md"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")

    x = np.asarray([rho for _, rho in RHO_SPECS])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for metric, label, color in (
        ("bacc", "BAcc", "#0072B2"), ("auc", "AUC", "#D55E00"),
        ("f1", "F1", "#009E73"),
    ):
        axes[0].errorbar(
            x, [r[f"selected_{metric}_mean"] for r in summaries],
            yerr=[r[f"selected_{metric}_sd"] for r in summaries],
            marker="o", linewidth=2, capsize=3, label=label, color=color,
        )
    axes[0].set_ylabel("Selected-checkpoint performance (%)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(
        x, [r["last20_auc_sd_mean"] for r in summaries],
        marker="s", linewidth=2, color="#CC79A7",
    )
    axes[1].set_ylabel("Mean last-20 AUC SD (points)")
    for axis, title in zip(axes, ("Predictive performance", "Late-round variation")):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(r"Positive allocation ratio $\rho$")
        axis.set_xticks(x)
        axis.grid(True, alpha=0.25)
    figure = output / "rho_sensitivity.png"
    fig.savefig(figure, dpi=300, bbox_inches="tight")
    fig.savefig(output / "rho_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Per-seed CSV -> {per_seed}")
    print(f"Summary CSV -> {summary_csv}")
    print(f"Paper table -> {table_path}")
    print(f"Figure -> {figure}")


if __name__ == "__main__":
    main()
