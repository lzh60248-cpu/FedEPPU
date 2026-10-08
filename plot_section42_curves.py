"""Create publication-style 5x3 per-round comparison figures for Section 4.2."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


DATASETS = ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")
METHODS = (
    "centralized_pn", "centralized_nnpu", "local_nnpu", "fedavg_nnpu",
    "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu",
)
SEEDS = (42, 43, 44, 45, 46)
METRICS = (("acc", "Accuracy"), ("auc", "AUC"), ("f1", "F1 score"))
DATASET_LABELS = {
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "chest": "Chest X-ray",
    "kolektor_sdd2": "KolektorSDD2",
}
METHOD_LABELS = {
    "centralized_pn": "Centralized PN",
    "centralized_nnpu": "Centralized nnPU",
    "local_nnpu": "Local nnPU",
    "fedavg_nnpu": "FedAvg-nnPU",
    "fedprox_nnpu": "FedProx-nnPU",
    "scaffold_nnpu": "SCAFFOLD-nnPU",
    "fedpu": "FedPU",
    "mrfl_nnpu": "MRFL-nnPU",
    "fedeppu": "FedEPPU (Main3)",
}
COLORS = {
    "centralized_pn": "#0072B2",
    "centralized_nnpu": "#56B4E9",
    "local_nnpu": "#999999",
    "fedavg_nnpu": "#E69F00",
    "fedprox_nnpu": "#D55E00",
    "scaffold_nnpu": "#CC79A7",
    "fedpu": "#F0E442",
    "mrfl_nnpu": "#009E73",
    "fedeppu": "#111111",
}
LINESTYLES = {
    "centralized_pn": "-", "centralized_nnpu": "--", "local_nnpu": ":",
    "fedavg_nnpu": "-", "fedprox_nnpu": "--", "scaffold_nnpu": "-.",
    "fedpu": ":", "mrfl_nnpu": "-.", "fedeppu": "-",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def result_path(root, dataset, method, seed):
    folder = root / dataset / method / f"seed{seed}"
    name = (f"fedeppu_{dataset}_seed{seed}.json" if method == "fedeppu"
            else f"{method}_{dataset}_seed{seed}.json")
    return folder / name


def load_history(root, dataset, method, seed):
    path = result_path(root, dataset, method, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing curve input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method == "fedeppu":
        source = payload["history"]
        curves = {metric: np.asarray(source[metric], dtype=float) for metric, _ in METRICS}
    else:
        source = payload.get("history", [])
        curves = {
            metric: np.asarray([row[metric] for row in source], dtype=float)
            for metric, _ in METRICS
        }
    for metric, curve in curves.items():
        if len(curve) != 200 or not np.isfinite(curve).all():
            raise ValueError(f"Invalid {metric} history in {path}: length={len(curve)}")
    return curves


def moving_average(values, window=5):
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def create_figure(root, dataset):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#B8B8B8",
        "xtick.color": "#4A4A4A",
        "ytick.color": "#4A4A4A",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(5, 3, figsize=(17.2, 18.5), sharex=True, sharey="col")
    rounds = np.arange(1, 201)
    cache = {
        (seed, method): load_history(root, dataset, method, seed)
        for seed in SEEDS for method in METHODS
    }
    for row, seed in enumerate(SEEDS):
        for column, (metric, title) in enumerate(METRICS):
            ax = axes[row, column]
            for method in METHODS:
                curve = cache[(seed, method)][metric]
                color = COLORS[method]
                # The faint trace preserves every observed round; the stronger
                # five-round mean makes overlapping methods readable.
                ax.plot(rounds, curve, color=color, linewidth=.45, alpha=.16)
                ax.plot(
                    rounds, moving_average(curve), color=color,
                    linestyle=LINESTYLES[method],
                    linewidth=2.15 if method == "fedeppu" else 1.25,
                    alpha=1.0 if method == "fedeppu" else .90,
                    zorder=5 if method == "fedeppu" else 3,
                )
            ax.set_xlim(1, 200)
            ax.set_ylim(0.40, 1.005)
            ax.set_xticks((1, 50, 100, 150, 200))
            ax.set_yticks(np.arange(.4, 1.01, .1))
            ax.grid(True, color="#D9D9D9", linewidth=.55, alpha=.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title, fontsize=12.5, pad=9)
            if column == 0:
                ax.set_ylabel(f"Seed {seed}\nMetric value", fontsize=10.5)
            if row == len(SEEDS) - 1:
                ax.set_xlabel("Communication round", fontsize=10.5)

    handles = [
        Line2D([0], [0], color=COLORS[method], linestyle=LINESTYLES[method],
               linewidth=2.5 if method == "fedeppu" else 1.7,
               label=METHOD_LABELS[method])
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .962), columnspacing=1.45, handlelength=3.1)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: per-round comparison across five seeds",
        fontsize=16, fontweight="bold", y=.993,
    )
    fig.text(
        .5, .012,
        "Faint lines show raw per-round values; solid traces show centered 5-round moving averages.",
        ha="center", va="bottom", fontsize=9, color="#555555",
    )
    fig.subplots_adjust(left=.075, right=.985, bottom=.045, top=.915, hspace=.24, wspace=.14)
    return fig


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "section42_all_datasets_curves.pdf"
    with PdfPages(combined_path) as pdf:
        for dataset in DATASETS:
            fig = create_figure(args.input_dir, dataset)
            png = args.output_dir / f"section42_{dataset}_five_seed_curves.png"
            single_pdf = args.output_dir / f"section42_{dataset}_five_seed_curves.pdf"
            fig.savefig(png, dpi=300, bbox_inches="tight")
            fig.savefig(single_pdf, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"figure={png}")
    print(f"combined_pdf={combined_path}")


if __name__ == "__main__":
    main()
