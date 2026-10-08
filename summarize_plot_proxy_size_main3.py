"""Validate, summarize, and plot the Main3 proxy-size sensitivity study."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROXY_SIZES = (128, 256, 512, 1024)
SEEDS = (42, 43, 44, 45, 46)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def find_latest_result(input_dir, proxy_size, seed):
    """Select the newest result for one setting across all retry job folders."""
    pattern = f"**/proxy{proxy_size}/seed{seed}/*.json"
    candidates = sorted(input_dir.glob(pattern))
    if not candidates:
        return None, 0
    return max(candidates, key=lambda path: path.stat().st_mtime), len(candidates)


def main():
    args = parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    reference_partitions = {}

    for proxy_size in PROXY_SIZES:
        for seed in SEEDS:
            result_file, candidate_count = find_latest_result(
                args.input_dir, proxy_size, seed
            )
            if result_file is None:
                missing.append(
                    f"proxy{proxy_size}/seed{seed}: found 0 JSON files"
                )
                continue
            print(
                f"selected_result=proxy{proxy_size}/seed{seed}; "
                f"candidates={candidate_count}; file={result_file}"
            )
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            history = payload["history"]
            selected = history.get("selected")
            if selected is None:
                raise ValueError(f"Selected checkpoint missing in {result_file}.")
            if len(history["auc"]) != 200:
                raise ValueError(f"Expected 200 rounds in {result_file}.")
            if int(metadata["effective_proxy_size"]) != proxy_size:
                raise ValueError(f"Wrong effective proxy size in {result_file}.")
            if int(metadata["reserved_proxy_pool_size"]) != 1024:
                raise ValueError(f"Wrong reserved proxy pool in {result_file}.")

            partition = (
                tuple(metadata["client_sample_counts"]),
                tuple(round(float(x), 12) for x in metadata["client_priors"]),
            )
            if seed not in reference_partitions:
                reference_partitions[seed] = partition
            elif partition != reference_partitions[seed]:
                raise ValueError(
                    f"Client partition changed with proxy size for seed {seed}."
                )

            records.append({
                "proxy_size": proxy_size,
                "seed": seed,
                "selection_size": int(metadata["proxy_selection_count"]),
                "calibration_size": int(metadata["proxy_calibration_count"]),
                "selected_round": int(selected["round"]),
                "bacc": float(selected["bacc"]) * 100.0,
                "auc": float(selected["auc"]) * 100.0,
                "threshold": float(selected["threshold"]),
                "source_file": str(result_file),
                "candidate_count": candidate_count,
            })

    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))
    if len(records) != len(PROXY_SIZES) * len(SEEDS):
        raise RuntimeError("The experiment must contain exactly 20 valid runs.")

    per_seed_path = output / "proxy_size_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for proxy_size in PROXY_SIZES:
        group = [row for row in records if row["proxy_size"] == proxy_size]
        summary = {
            "proxy_size": proxy_size,
            "selection_size_min": min(row["selection_size"] for row in group),
            "selection_size_max": max(row["selection_size"] for row in group),
            "calibration_size_min": min(row["calibration_size"] for row in group),
            "calibration_size_max": max(row["calibration_size"] for row in group),
        }
        for field in ("selected_round", "bacc", "auc", "threshold"):
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(
                [row[field] for row in group]
            )
        summaries.append(summary)

    summary_path = output / "proxy_size_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    def range_cell(low, high):
        return str(low) if low == high else f"{low}--{high}"

    table = [
        "| Proxy samples | Selection samples | Calibration samples | Selected round | BAcc (%) | AUC (%) | Threshold SD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append("| " + " | ".join([
            str(row["proxy_size"]),
            range_cell(row["selection_size_min"], row["selection_size_max"]),
            range_cell(row["calibration_size_min"], row["calibration_size_max"]),
            f'{row["selected_round_mean"]:.1f} ± {row["selected_round_sd"]:.1f}',
            f'{row["bacc_mean"]:.2f} ± {row["bacc_sd"]:.2f}',
            f'{row["auc_mean"]:.2f} ± {row["auc_sd"]:.2f}',
            f'{row["threshold_sd"]:.3f}',
        ]) + " |")
    table_path = output / "proxy_size_table.md"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")

    # Treat the four prescribed proxy budgets as ordered experimental settings.
    # Equal spacing makes the two-fold increase at every setting easy to read,
    # while avoiding the visual crowding caused by a narrow log-scale panel.
    x = np.arange(len(PROXY_SIZES), dtype=float)
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.28, 1.0)},
    )
    for field, label, color in (
        ("bacc", "BAcc", "#0072B2"),
        ("auc", "AUC", "#D55E00"),
    ):
        axes[0].errorbar(
            x,
            [row[f"{field}_mean"] for row in summaries],
            yerr=[row[f"{field}_sd"] for row in summaries],
            marker="o", markersize=6.5, linewidth=2.35, capsize=4,
            elinewidth=1.7, label=label, color=color,
        )
    axes[0].set_ylabel("Selected-checkpoint performance (%)")
    axes[0].set_title("(a) Final performance", loc="left", fontweight="bold")

    threshold_sd = [row["threshold_sd"] for row in summaries]
    bars = axes[1].bar(
        x, threshold_sd, width=0.58, color="#009E73", edgecolor="#006B58",
        linewidth=0.8,
    )
    for bar, value in zip(bars, threshold_sd):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
            f"{value:.2f}", ha="center", va="bottom", fontsize=9,
            color="#006B58", clip_on=False,
        )
    axes[1].set_ylabel("Threshold SD across five seeds")
    axes[1].set_title("(b) Calibration-threshold variability", loc="left",
                      fontweight="bold")
    axes[1].set_ylim(0, max(threshold_sd) * 1.24 if max(threshold_sd) else 1.0)
    for axis in axes:
        axis.set_xticks(x, labels=[str(v) for v in PROXY_SIZES])
        axis.set_xlabel("Number of proxy samples")
        axis.grid(axis="y", alpha=0.24)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04), columnspacing=2.8)
    figure_png = output / "proxy_size_sensitivity.png"
    figure_pdf = output / "proxy_size_sensitivity.pdf"
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"validated_results={len(records)}")
    print("client_partition_check=identical_across_proxy_sizes_within_each_seed")
    print(f"per_seed={per_seed_path}")
    print(f"summary={summary_path}")
    print(f"table={table_path}")
    print(f"figure_png={figure_png}")
    print(f"figure_pdf={figure_pdf}")


if __name__ == "__main__":
    main()
