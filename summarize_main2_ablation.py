"""Summarize five-seed Main2 ablations and their percentage-point changes."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("main2_ablation_outputs")
VARIANTS = ("main2_full", "no_ranking", "fedavg")
METRICS = ("acc", "f1", "bacc", "auc")


def load_records():
    records = {variant: [] for variant in VARIANTS}
    for path in ROOT.glob("*/seed*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = payload["metadata"]["variant"]
        if variant in records:
            records[variant].append(payload["history"]["selected"])
    missing = {name: 5 - len(rows) for name, rows in records.items() if len(rows) != 5}
    if missing:
        raise SystemExit(f"Expected exactly five runs per variant; incomplete={missing}")
    return records


def metric_stats(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=float) * 100.0
    return float(values.mean()), float(values.std(ddof=1))


def main():
    records = load_records()
    means = {}
    summary_path = ROOT / "main2_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        header = ["variant", "runs"]
        for metric in METRICS:
            header.extend((f"{metric}_mean_pct", f"{metric}_std_pct"))
        writer.writerow(header)
        for variant in VARIANTS:
            row = [variant, len(records[variant])]
            means[variant] = {}
            for metric in METRICS:
                mean, std = metric_stats(records[variant], metric)
                means[variant][metric] = mean
                row.extend((f"{mean:.2f}", f"{std:.2f}"))
            writer.writerow(row)

    difference_path = ROOT / "main2_ablation_differences.csv"
    labels = {
        "no_ranking": "Main2 full - no PU ranking",
        "fedavg": "Main2 full - FedAvg aggregation",
    }
    with difference_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("comparison", "acc_change_pp", "f1_change_pp", "bacc_change_pp", "auc_change_pp"))
        for variant in ("no_ranking", "fedavg"):
            changes = [means["main2_full"][metric] - means[variant][metric] for metric in METRICS]
            writer.writerow((labels[variant], *(f"{value:+.2f}" for value in changes)))

    print(f"summary={summary_path}")
    print(f"differences={difference_path}")


if __name__ == "__main__":
    main()
