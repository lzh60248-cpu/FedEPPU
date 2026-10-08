"""Plot CIFAR-10 seed-42 per-round ACC, AUC and F1 for all Section 4.2 methods."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHODS = (
    "centralized_pn", "centralized_nnpu", "local_nnpu", "fedavg_nnpu",
    "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu",
)
LABELS = {
    "centralized_pn": "Centralized PN",
    "centralized_nnpu": "Centralized nnPU",
    "local_nnpu": "Local nnPU",
    "fedavg_nnpu": "FedAvg-nnPU",
    "fedprox_nnpu": "FedProx-nnPU",
    "scaffold_nnpu": "SCAFFOLD-nnPU",
    "fedpu": "FedPU",
    "mrfl_nnpu": "MRFL-nnPU",
    "fedeppu": "FedEPPU",
}
COLORS = {
    "centralized_pn": "#0072B2",
    "centralized_nnpu": "#56B4E9",
    "local_nnpu": "#8C8C8C",
    "fedavg_nnpu": "#E69F00",
    "fedprox_nnpu": "#D55E00",
    "scaffold_nnpu": "#CC79A7",
    "fedpu": "#F0C808",
    "mrfl_nnpu": "#009E73",
    "fedeppu": "#111111",
}
LINESTYLES = {
    "centralized_pn": "-", "centralized_nnpu": "--", "local_nnpu": ":",
    "fedavg_nnpu": "-", "fedprox_nnpu": "--", "scaffold_nnpu": "-.",
    "fedpu": ":", "mrfl_nnpu": "-.", "fedeppu": "-",
}
METRICS = (("acc", "Accuracy"), ("auc", "AUC"), ("f1", "F1 score"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("section42_outputs"))
    parser.add_argument("--output_dir", type=Path, default=Path("section42_seed42_figure"))
    return parser.parse_args()


def job_number(path, root):
    first = path.relative_to(root).parts[0]
    try:
        return int(first)
    except ValueError as error:
        raise ValueError(f"Expected numeric job directory, found {first!r} in {path}") from error


def find_latest(root, method):
    candidates = list(root.glob(
        f"*/cifar10/{method}/seed42/{method}_cifar10_seed42.json"
    ))
    if not candidates:
        raise FileNotFoundError(
            f"Missing CIFAR-10 seed42 JSON for {method} under {root}"
        )
    selected = max(candidates, key=lambda path: job_number(path, root))
    print(
        f"selected method={method}; job={job_number(selected, root)}; file={selected}",
        flush=True,
    )
    return selected


def load_curves(path, method):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method == "fedeppu":
        history = payload["history"]
        curves = {metric: np.asarray(history[metric], dtype=float)
                  for metric, _ in METRICS}
    else:
        history = payload.get("history", [])
        curves = {
            metric: np.asarray([row[metric] for row in history], dtype=float)
            for metric, _ in METRICS
        }
    for metric, values in curves.items():
        if len(values) != 200:
            raise ValueError(f"{path}: expected 200 {metric} values, found {len(values)}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite values in {metric}")
    return curves


def moving_average(values, window=5):
    left = window // 2
    right = window - left - 1
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = {
        method: load_curves(find_latest(args.input_dir, method), method)
        for method in METHODS
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#B8B8B8",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    })
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5), sharex=True)
    rounds = np.arange(1, 201)
    for axis, (metric, title) in zip(axes, METRICS):
        for method in METHODS:
            values = curves[method][metric]
            axis.plot(
                rounds, values, color=COLORS[method], linewidth=.5,
                alpha=.15, zorder=1,
            )
            axis.plot(
                rounds, moving_average(values), color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.5 if method == "fedeppu" else 1.45,
                alpha=1.0 if method == "fedeppu" else .92,
                zorder=5 if method == "fedeppu" else 3,
            )
        axis.set_title(title, fontsize=13, pad=9)
        axis.set_xlabel("Communication round")
        axis.set_xlim(1, 200)
        axis.set_ylim(0.0, 1.01)
        axis.set_xticks((1, 50, 100, 150, 200))
        axis.set_yticks(np.arange(0, 1.01, .1))
        axis.grid(True, color="#D9D9D9", linewidth=.55, alpha=.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Metric value")

    handles = [
        Line2D([0], [0], color=COLORS[method], linestyle=LINESTYLES[method],
               linewidth=2.7 if method == "fedeppu" else 1.8,
               label=LABELS[method])
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .965), handlelength=3.0, columnspacing=1.4)
    fig.suptitle("CIFAR-10, seed 42: per-round comparison", fontsize=16,
                 fontweight="bold", y=1.04)
    fig.text(
        .5, .015,
        "Faint lines: raw per-round values. Strong lines: centered 5-round moving averages.",
        ha="center", color="#555555", fontsize=9,
    )
    fig.subplots_adjust(left=.055, right=.99, bottom=.14, top=.82, wspace=.17)

    png = args.output_dir / "section42_cifar10_seed42_acc_auc_f1.png"
    pdf = args.output_dir / "section42_cifar10_seed42_acc_auc_f1.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"png={png.resolve()}")
    print(f"pdf={pdf.resolve()}")


if __name__ == "__main__":
    main()
