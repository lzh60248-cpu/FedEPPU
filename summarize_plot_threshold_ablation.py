"""Summarize and plot the paired Main3 threshold-calibration ablation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44, 45, 46)
PROTOCOLS = ("calibrated", "zero")
LABELS = {
    "calibrated": "Proxy-calibrated threshold",
    "zero": "Fixed zero threshold",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def sample_sd(values):
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def load_seed(root, seed):
    path = root / f"seed{seed}" / f"threshold_cifar10_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing threshold result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload["history"]
    if len(history["bacc"]) != 200 or len(history["zero_bacc"]) != 200:
        raise ValueError(f"Expected 200 paired rounds in {path}")
    return {
        "calibrated": {
            "selected": history["selected"],
            "bacc": np.asarray(history["bacc"], dtype=float) * 100.0,
        },
        "zero": {
            "selected": history["selected_zero"],
            "bacc": np.asarray(history["zero_bacc"], dtype=float) * 100.0,
        },
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_data = {seed: load_seed(args.input_dir, seed) for seed in SEEDS}
    summary = {}
    rows = []
    for protocol in PROTOCOLS:
        selected = [seed_data[seed][protocol]["selected"] for seed in SEEDS]
        curves = [seed_data[seed][protocol]["bacc"] for seed in SEEDS]
        row = {"protocol": protocol, "label": LABELS[protocol]}
        for metric in ("acc", "f1", "bacc"):
            values = np.asarray([float(item[metric]) for item in selected]) * 100.0
            row[f"final_{metric}_mean"] = values.mean()
            row[f"final_{metric}_sd"] = values.std(ddof=1)
        last20_mean = np.asarray([curve[-20:].mean() for curve in curves])
        temporal_sd = np.asarray([curve[-20:].std(ddof=1) for curve in curves])
        temporal_range = np.asarray([np.ptp(curve[-20:]) for curve in curves])
        row["last20_bacc_mean"] = last20_mean.mean()
        row["last20_bacc_seed_sd"] = last20_mean.std(ddof=1)
        row["last20_temporal_sd_mean"] = temporal_sd.mean()
        row["last20_temporal_sd_seed_sd"] = temporal_sd.std(ddof=1)
        row["last20_range_mean"] = temporal_range.mean()
        row["last20_range_seed_sd"] = temporal_range.std(ddof=1)
        rows.append(row)
        summary[protocol] = row

    with (args.output_dir / "threshold_ablation_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "| Inference protocol | Final ACC (%) | Final F1 (%) | Final BAcc (%) | Last-20 BAcc (%) | Last-20 BAcc SD | Last-20 BAcc range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [
            row["label"],
            f"{row['final_acc_mean']:.2f} ± {row['final_acc_sd']:.2f}",
            f"{row['final_f1_mean']:.2f} ± {row['final_f1_sd']:.2f}",
            f"{row['final_bacc_mean']:.2f} ± {row['final_bacc_sd']:.2f}",
            f"{row['last20_bacc_mean']:.2f} ± {row['last20_bacc_seed_sd']:.2f}",
            f"{row['last20_temporal_sd_mean']:.2f} ± {row['last20_temporal_sd_seed_sd']:.2f}",
            f"{row['last20_range_mean']:.2f} ± {row['last20_range_seed_sd']:.2f}",
        ]
        table.append("| " + " | ".join(cells) + " |")
    (args.output_dir / "threshold_ablation_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#B8B8B8",
    })
    fig, axis = plt.subplots(figsize=(10.6, 5.7))
    rounds = np.arange(1, 201)
    styles = {
        "calibrated": ("#111111", "-"),
        "zero": ("#D55E00", "--"),
    }
    for protocol in PROTOCOLS:
        matrix = np.stack([seed_data[seed][protocol]["bacc"] for seed in SEEDS])
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        color, linestyle = styles[protocol]
        axis.plot(rounds, mean, color=color, linestyle=linestyle,
                  linewidth=2.4, label=LABELS[protocol])
        axis.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=.13)
    axis.set_title("Threshold calibration: CIFAR-10 balanced accuracy", fontweight="bold")
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Balanced accuracy (%)")
    axis.set_xlim(1, 200)
    axis.grid(True, color="#D9D9D9", linewidth=.6, alpha=.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    annotation = (
        "Last 20 rounds\n"
        f"Calibrated: {summary['calibrated']['last20_bacc_mean']:.2f}%  "
        f"(SD {summary['calibrated']['last20_temporal_sd_mean']:.2f})\n"
        f"Zero: {summary['zero']['last20_bacc_mean']:.2f}%  "
        f"(SD {summary['zero']['last20_temporal_sd_mean']:.2f})"
    )
    axis.text(.02, .04, annotation, transform=axis.transAxes, va="bottom",
              bbox=dict(boxstyle="round,pad=.45", facecolor="white",
                        edgecolor="#C8C8C8", alpha=.92))
    fig.tight_layout()
    png = args.output_dir / "threshold_calibration_bacc_curve.png"
    pdf = args.output_dir / "threshold_calibration_bacc_curve.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("validated_paired_runs=5")
    print(f"table={args.output_dir / 'threshold_ablation_table.md'}")
    print(f"figure={png}")


if __name__ == "__main__":
    main()
