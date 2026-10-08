"""Collect complete Main3 step-cap runs and render the paper's Figure 17.

The collector deliberately selects the newest JSON for each (step cap, seed)
combination.  It therefore also works when an interrupted Slurm array is
resubmitted and its completed elements are split across several array jobs.
"""

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_CAPS = (5, 10, 15, 20)
SEEDS = (42, 43, 44, 45, 46)


def sample_sd(values):
    return float(np.asarray(values, dtype=float).std(ddof=1))


def latest_source(input_root, step_cap, seed):
    candidates = list(input_root.glob(
        f"*/steps{step_cap}/seed{seed}/*.json"
    ))
    if not candidates:
        raise FileNotFoundError(
            f"Missing result: steps{step_cap}/seed{seed} under {input_root}"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def extract_metrics(payload):
    history = payload.get("history", {})
    selected = history.get("selected")
    if not isinstance(selected, dict) or "auc" not in selected:
        raise ValueError("The result has no selected-checkpoint AUC")
    auc_curve = np.asarray(history.get("auc", []), dtype=float) * 100.0
    if auc_curve.size < 20:
        raise ValueError("The result has fewer than 20 communication rounds")
    return {
        "selected_round": int(selected["round"]),
        "final_auc": float(selected["auc"]) * 100.0,
        "last20_auc_sd": sample_sd(auc_curve[-20:]),
    }


def write_csv(path, rows, fields):
    import csv
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("step_trend_main3_outputs"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("figure17_step_sensitivity_artifacts"))
    args = parser.parse_args()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Missing step-trend root: {args.input_root}")
    output = args.output_root
    merged = output / "merged_results"
    output.mkdir(parents=True, exist_ok=True)

    per_seed, manifest = [], []
    for step_cap in STEP_CAPS:
        for seed in SEEDS:
            source = latest_source(args.input_root, step_cap, seed)
            payload = json.loads(source.read_text(encoding="utf-8"))
            metrics = extract_metrics(payload)
            target = merged / f"steps{step_cap}" / f"seed{seed}" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            per_seed.append({"step_cap": step_cap, "seed": seed, **metrics})
            manifest.append(
                f"steps{step_cap},seed{seed},source={source.resolve()}"
            )

    write_csv(output / "figure17_per_seed.csv", per_seed,
              ["step_cap", "seed", "selected_round", "final_auc", "last20_auc_sd"])
    (output / "figure17_merge_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

    summary = []
    for step_cap in STEP_CAPS:
        group = [row for row in per_seed if row["step_cap"] == step_cap]
        auc = [row["final_auc"] for row in group]
        late_sd = [row["last20_auc_sd"] for row in group]
        summary.append({
            "step_cap": step_cap,
            "final_auc_mean": float(np.mean(auc)),
            "final_auc_sample_sd": sample_sd(auc),
            "last20_auc_sd_mean": float(np.mean(late_sd)),
            "last20_auc_sd_sample_sd": sample_sd(late_sd),
        })
    write_csv(output / "figure17_summary.csv", summary, list(summary[0]))

    table = [
        "| Maximum local steps | Final AUC (%) | Last-20 AUC SD (points) |",
        "|---:|---:|---:|",
    ]
    for row in summary:
        table.append(
            f'| {row["step_cap"]} | '
            f'{row["final_auc_mean"]:.2f} ± {row["final_auc_sample_sd"]:.2f} | '
            f'{row["last20_auc_sd_mean"]:.3f} ± '
            f'{row["last20_auc_sd_sample_sd"]:.3f} |'
        )
    (output / "figure17_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    x = np.asarray(STEP_CAPS, dtype=float)
    fig, left = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    right = left.twinx()
    left.errorbar(
        x, [row["final_auc_mean"] for row in summary],
        yerr=[row["final_auc_sample_sd"] for row in summary],
        color="#0072B2", marker="o", markersize=6, linewidth=2.3,
        capsize=4, label="Final AUC",
    )
    right.errorbar(
        x, [row["last20_auc_sd_mean"] for row in summary],
        yerr=[row["last20_auc_sd_sample_sd"] for row in summary],
        color="#D55E00", marker="s", markersize=5.5, linewidth=2.3,
        capsize=4, label="Last-20 AUC SD",
    )
    left.set_xlabel("Maximum local updates per epoch")
    left.set_ylabel("Selected-checkpoint AUC (%)", color="#0072B2")
    right.set_ylabel("Last-20-round AUC SD (points)", color="#D55E00")
    left.tick_params(axis="y", colors="#0072B2")
    right.tick_params(axis="y", colors="#D55E00")
    left.set_xticks(x)
    left.grid(axis="both", alpha=0.22)
    handles = left.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = left.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06), columnspacing=2.5)
    fig.savefig(output / "figure17_step_sensitivity.png", dpi=350,
                bbox_inches="tight")
    fig.savefig(output / "figure17_step_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("validated_results=20")
    print(f"artifacts={output.resolve()}")


if __name__ == "__main__":
    main()
