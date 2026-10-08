"""Create the publication figure for Main3 local classification-ranking losses."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SEEDS = (42, 43, 44, 45, 46)
SERIES = (
    ("local_classification_loss", r"(a) nnPU classification loss $L_c$", "#2166AC"),
    ("local_ranking_loss", r"(b) PU ranking loss $L_r$", "#B2182B"),
    ("local_joint_loss", r"(c) Joint loss $L_c+L_r$", "#5E3C99"),
)


def trailing_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window > values.size:
        raise ValueError("moving-average window must be between 1 and run length")
    cumulative = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def load_runs(root: Path, rounds: int):
    runs = {}
    for path in root.glob("**/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if metadata.get("method") != "FedEPPU-Main3-LocalLossCurves":
            continue
        seed = int(metadata["args"]["seed"])
        history = payload["history"]
        if seed in runs:
            raise ValueError(f"duplicate seed {seed}: {path}")
        for key, _, _ in SERIES:
            if len(history.get(key, [])) != rounds:
                raise ValueError(f"{path}: {key} does not contain {rounds} rounds")
        runs[seed] = history
    missing = sorted(set(EXPECTED_SEEDS) - set(runs))
    extra = sorted(set(runs) - set(EXPECTED_SEEDS))
    if missing or extra:
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}; missing={missing}; extra={extra}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("main3_local_loss_outputs"))
    parser.add_argument("--output", type=Path,
                        default=Path("main3_local_loss_outputs/figure15_local_losses.pdf"))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--window", type=int, default=10,
                        help="Trailing moving-average window used only for visualization.")
    args = parser.parse_args()

    runs = load_runs(args.root, args.rounds)
    rounds = np.arange(args.window, args.rounds + 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)
    summary_rows = []

    for axis, (key, title, color) in zip(axes, SERIES):
        matrix = np.stack([
            trailing_moving_average(np.asarray(runs[seed][key]), args.window)
            for seed in EXPECTED_SEEDS
        ])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        axis.fill_between(rounds, mean - std, mean + std, color=color,
                          alpha=0.18, linewidth=0, label=r"Mean $\pm$ SD")
        axis.plot(rounds, mean, color=color, linewidth=1.7, label="Mean")
        axis.set_title(title, pad=7)
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Loss")
        axis.set_xlim(args.window, args.rounds)
        axis.grid(True, which="major", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3, width=0.7)
        for round_number, mean_value, std_value in zip(rounds, mean, std):
            summary_rows.append((key, round_number, mean_value, std_value))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=2, frameon=False, handlelength=2.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("series", "round", "mean", "sample_std"))
        writer.writerows(summary_rows)
    print(f"figure_pdf={args.output}")
    print(f"figure_png={png_path}")
    print(f"figure_data={csv_path}")
    print(f"seeds={EXPECTED_SEEDS}; moving_average_window={args.window}; ddof=1")


if __name__ == "__main__":
    main()
