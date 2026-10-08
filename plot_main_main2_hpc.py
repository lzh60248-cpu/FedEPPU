"""Plot five-seed per-round BAcc and AUC for the current Main and Main2."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44, 45, 46)
METHODS = {
    "main": {"label": "Main", "color": "#D55E00"},
    "main2": {"label": "Main2", "color": "#0072B2"},
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window > len(values):
        raise ValueError("window must be between 1 and the number of rounds")
    if window == 1:
        return values.astype(np.float64, copy=True)
    padded = np.pad(values.astype(np.float64), (window - 1, 0), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def load_histories(root: Path, rounds: int):
    histories = {method: {} for method in METHODS}
    for method in METHODS:
        for seed in SEEDS:
            candidates = list((root / method / f"seed{seed}").glob("*.json"))
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one JSON for {method} seed {seed}, found {len(candidates)}"
                )
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            history = payload["history"]
            for metric in ("bacc", "auc"):
                if len(history.get(metric, [])) != rounds:
                    raise ValueError(
                        f"{candidates[0]}: {metric} has {len(history.get(metric, []))} rounds"
                    )
            histories[method][seed] = history
    return histories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--window", type=int, default=5,
                        help="Moving-average window used only for visualization.")
    args = parser.parse_args()
    histories = load_histories(args.root, args.rounds)
    x = np.arange(1, args.rounds + 1)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    metrics = (("bacc", "(a) Balanced Accuracy", "BAcc"),
               ("auc", "(b) Area Under the ROC Curve", "AUC"))
    table_rows = []

    for axis, (metric, title, ylabel) in zip(axes, metrics):
        for method, style in METHODS.items():
            matrix = np.stack([
                moving_average(np.asarray(histories[method][seed][metric]), args.window)
                for seed in SEEDS
            ])
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1)
            for row in matrix:
                axis.plot(x, row, color=style["color"], linewidth=.65, alpha=.16)
            axis.fill_between(x, mean - sd, mean + sd, color=style["color"],
                              alpha=.15, linewidth=0)
            axis.plot(x, mean, color=style["color"], linewidth=1.8,
                      label=style["label"] + r" mean $\pm$ SD")
            for round_number, mean_value, sd_value in zip(x, mean, sd):
                table_rows.append((method, metric, round_number, mean_value, sd_value))
        axis.set_title(title, pad=6)
        axis.set_xlabel("Communication round")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, args.rounds)
        axis.grid(True, color="#D9D9D9", linewidth=.5, alpha=.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "metric", "round", "mean", "sample_std"))
        writer.writerows(table_rows)
    print(f"figure_pdf={args.output}")
    print(f"figure_png={png_path}")
    print(f"figure_data={csv_path}")
    print(f"seeds={SEEDS}; rounds={args.rounds}; moving_average_window={args.window}")


if __name__ == "__main__":
    main()
