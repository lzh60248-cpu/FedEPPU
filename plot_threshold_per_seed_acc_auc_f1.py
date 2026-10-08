"""Create one ACC/AUC/F1 threshold-comparison figure for each CIFAR-10 seed."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44, 45, 46)
METRICS = (("acc", "Accuracy"), ("auc", "AUC"), ("f1", "F1 score"))
COLORS = {"calibrated": "#111111", "zero": "#D55E00"}
LABELS = {"calibrated": "Proxy-calibrated threshold", "zero": "Fixed zero threshold"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("threshold_ablation_outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("threshold_ablation_artifacts"))
    return parser.parse_args()


def result_path(job_dir, seed):
    return job_dir / f"seed{seed}" / f"threshold_cifar10_seed{seed}.json"


def complete_job_dirs(input_root):
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing result root: {input_root}")
    return sorted(
        [directory for directory in input_root.iterdir() if directory.is_dir()
         and all(result_path(directory, seed).is_file() for seed in SEEDS)],
        key=lambda directory: directory.name,
    )


def load_seed(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", {})
    result = {}
    for metric, _ in METRICS:
        calibrated = np.asarray(history.get(metric, []), dtype=float)
        zero = np.asarray(history.get(f"zero_{metric}", []), dtype=float)
        if calibrated.shape != (200,) or zero.shape != (200,):
            raise ValueError(f"Expected 200 paired {metric} values in {path}")
        result[metric] = (calibrated * 100.0, zero * 100.0)
    return result


def draw_seed(seed, data, output):
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6), sharex=True)
    rounds = np.arange(1, 201)
    for axis, (metric, title) in zip(axes, METRICS):
        calibrated, zero = data[metric]
        axis.plot(rounds, calibrated, color=COLORS["calibrated"], linewidth=2.2,
                  label=LABELS["calibrated"])
        axis.plot(rounds, zero, color=COLORS["zero"], linewidth=1.8,
                  linestyle="--", label=LABELS["zero"])
        axis.set_title(title)
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Score (%)")
        axis.set_xlim(1, 200)
        axis.grid(True, color="#D9D9D9", linewidth=.65, alpha=.85)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.07))
    fig.suptitle(f"CIFAR-10 threshold calibration ablation (seed {seed})", y=1.01,
                 fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .87))
    stem = output / f"threshold_acc_auc_f1_seed{seed}"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    job_dirs = complete_job_dirs(args.input_root)
    if not job_dirs:
        raise FileNotFoundError(
            "No complete threshold-ablation result set found; expected five JSON "
            f"files under {args.input_root}/<job-id>/seedXX/."
        )
    source = job_dirs[-1]
    output = args.output_dir / source.name
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        draw_seed(seed, load_seed(result_path(source, seed)), output)
    print(f"source_job={source.name}")
    print("validated_paired_runs=5")
    print(f"artifacts={output.resolve()}")


if __name__ == "__main__":
    main()
