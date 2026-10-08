"""Render five CIFAR-10 per-seed federated-method trajectory figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

SEEDS = (42, 43, 44, 45, 46)
METHODS = ("fedavg_nnpu", "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu")
METRICS = (("acc", "ACC"), ("auc", "AUC"), ("f1", "F1"))
METHOD_LABELS = {"fedavg_nnpu": "FedAvg-nnPU", "fedprox_nnpu": "FedProx-nnPU",
                 "scaffold_nnpu": "SCAFFOLD-nnPU", "fedpu": "FedPU",
                 "mrfl_nnpu": "MRFL-nnPU", "fedeppu": "FedEPPU"}
COLORS = {"fedavg_nnpu": "#0072B2", "fedprox_nnpu": "#D55E00",
          "scaffold_nnpu": "#CC79A7", "fedpu": "#B57600",
          "mrfl_nnpu": "#009E73", "fedeppu": "#202020"}
LINESTYLES = {"fedavg_nnpu": "-", "fedprox_nnpu": "--", "scaffold_nnpu": "-.",
              "fedpu": ":", "mrfl_nnpu": (0, (5, 1.2)), "fedeppu": "-"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("section42_paper_artifacts/merged_results"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figure1_cifar10_federated_seed_artifacts"))
    return parser.parse_args()


def load_history(root, method, seed):
    filename = f"fedeppu_cifar10_seed{seed}.json" if method == "fedeppu" else f"{method}_cifar10_seed{seed}.json"
    path = root / "cifar10" / method / f"seed{seed}" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIFAR-10 curve: {path}")
    history = json.loads(path.read_text(encoding="utf-8")).get("history")
    if method == "fedeppu":
        curves = {key: np.asarray(history.get(key, []), dtype=float) for key, _ in METRICS}
    else:
        curves = {key: np.asarray([row[key] for row in history], dtype=float) for key, _ in METRICS}
    for key, curve in curves.items():
        if curve.size != 200 or not np.isfinite(curve).all():
            raise ValueError(f"Invalid {key} trajectory in {path}")
    return curves


def moving_average(values, window=5):
    left, right = window // 2, window - 1 - window // 2
    return np.convolve(np.pad(values, (left, right), mode="edge"), np.ones(window) / window, mode="valid")


def create_figure(histories, seed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#A8A8A8", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.65), sharex=True, sharey=True,
                                constrained_layout=False)
    rounds = np.arange(1, 201)
    for axis, (metric, metric_label) in zip(axes, METRICS):
        for method in METHODS:
            curve = histories[method][metric] * 100.0
            axis.plot(rounds, curve, color=COLORS[method], linewidth=.45, alpha=.14)
            axis.plot(rounds, moving_average(curve), color=COLORS[method],
                      linestyle=LINESTYLES[method], linewidth=2.35 if method == "fedeppu" else 1.40,
                      alpha=1.0 if method == "fedeppu" else .94, zorder=5 if method == "fedeppu" else 3)
        axis.set_title(metric_label, fontweight="bold", pad=8)
        axis.set_xlim(1, 200); axis.set_ylim(45, 100.5)
        axis.set_xticks((1, 50, 100, 150, 200)); axis.set_yticks((50, 60, 70, 80, 90, 100))
        axis.set_xlabel("Communication round")
        axis.grid(axis="both", color="#D9D9D9", linewidth=.6, alpha=.72)
        axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Metric value (%)")
    handles = [Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m],
                      linewidth=2.5 if m == "fedeppu" else 1.8, label=METHOD_LABELS[m]) for m in METHODS]
    figure.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
                  bbox_to_anchor=(.5, .995), columnspacing=1.65, handlelength=2.7)
    figure.suptitle(f"CIFAR-10: federated-method trajectories (seed {seed})", fontsize=14.5, fontweight="bold", y=.885)
    figure.subplots_adjust(left=.062, right=.992, bottom=.115, top=.735, wspace=.085)
    return figure


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Missing merged Section 4.2 results: {args.input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "figure1_cifar10_federated_seed42_46.pdf"
    with PdfPages(combined) as output_pdf:
        for seed in SEEDS:
            histories = {method: load_history(args.input_root, method, seed) for method in METHODS}
            figure = create_figure(histories, seed)
            png = args.output_dir / f"figure1_cifar10_federated_seed{seed}.png"
            pdf = args.output_dir / f"figure1_cifar10_federated_seed{seed}.pdf"
            figure.savefig(png, dpi=350, bbox_inches="tight")
            figure.savefig(pdf, bbox_inches="tight")
            output_pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            print(f"figure={png}")
    print("validated_histories=30")
    print(f"combined_pdf={combined}")


if __name__ == "__main__":
    main()
