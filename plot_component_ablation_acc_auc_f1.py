"""Draw ACC, AUC and F1 trajectories for the five CIFAR-10 ablations.

The figure is created from completed JSON histories only.  It does not launch
training and it automatically selects the latest complete 25-run ablation set.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = (
    "full", "no_ranking", "no_complement", "no_support", "fedavg_aggregation",
)
SEEDS = (42, 43, 44, 45, 46)
LABELS = {
    "full": "FedEPPU",
    "no_ranking": "w/o PU ranking",
    "no_complement": "w/o complementary mass",
    "no_support": "w/o negative-support correction",
    "fedavg_aggregation": "FedAvg aggregation",
}
COLORS = {
    "full": "#111111",
    "no_ranking": "#0072B2",
    "no_complement": "#E69F00",
    "no_support": "#009E73",
    "fedavg_aggregation": "#CC79A7",
}
METRICS = (("acc", "Accuracy"), ("auc", "AUC"), ("f1", "F1 score"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("component_ablation_outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("component_ablation_artifacts"))
    return parser.parse_args()


def result_path(job_dir: Path, variant: str, seed: int) -> Path:
    return job_dir / variant / f"seed{seed}" / f"{variant}_cifar10_seed{seed}.json"


def complete_job_dirs(input_root: Path):
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing result root: {input_root}")
    complete = []
    for job_dir in input_root.iterdir():
        if not job_dir.is_dir():
            continue
        if all(result_path(job_dir, variant, seed).is_file()
               for variant in VARIANTS for seed in SEEDS):
            complete.append(job_dir)
    return sorted(complete, key=lambda path: path.name)


def load_curve(path: Path, metric: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", {})
    curve = np.asarray(history.get(metric, []), dtype=float)
    if curve.shape != (200,) or not np.all(np.isfinite(curve)):
        raise ValueError(f"Expected 200 finite {metric} values in {path}")
    return curve * 100.0


def main():
    args = parse_args()
    job_dirs = complete_job_dirs(args.input_root)
    if not job_dirs:
        raise FileNotFoundError(
            "No complete ablation set found; expected all 25 JSON files under "
            f"{args.input_root}/<job-id>/"
        )
    job_dir = job_dirs[-1]
    output = args.output_dir / job_dir.name
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.7), sharex=True)
    rounds = np.arange(1, 201)
    for axis, (metric, title) in zip(axes, METRICS):
        for variant in VARIANTS:
            curves = np.stack([
                load_curve(result_path(job_dir, variant, seed), metric)
                for seed in SEEDS
            ])
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1)
            axis.plot(rounds, mean, color=COLORS[variant],
                      linewidth=2.5 if variant == "full" else 1.65,
                      label=LABELS[variant])
            axis.fill_between(rounds, mean - sd, mean + sd,
                              color=COLORS[variant], alpha=0.10, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Score (%)")
        axis.set_xlim(1, 200)
        axis.grid(True, color="#D9D9D9", linewidth=.65, alpha=.85)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, .88))
    for suffix, kwargs in (("png", {"dpi": 350}), ("pdf", {})):
        fig.savefig(output / f"component_ablation_acc_auc_f1_curves.{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"source_job={job_dir.name}")
    print("validated_results=25")
    print(f"artifacts={output.resolve()}")


if __name__ == "__main__":
    main()
