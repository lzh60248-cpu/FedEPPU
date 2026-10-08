"""Create the five-seed comparison table for Main2 and independent baselines."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("comparison_main2_outputs/cifar10")
METHODS = (
    "centralized_pn", "centralized_nnpu", "local_nnpu", "fedavg_nnpu",
    "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu",
)
NON_CURVE_METHODS = {"centralized_pn", "centralized_nnpu", "local_nnpu"}


def load_run(path, method):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method == "fedeppu":
        history = payload["history"]
        selected = history["selected"]
        final = {"bacc": selected["bacc"], "auc": selected["auc"]}
        auc_curve = np.asarray(history["auc"], dtype=float)
        seed = int(payload["metadata"]["args"]["seed"])
    else:
        final = payload["metrics"]
        auc_curve = np.asarray(
            [row["auc"] for row in payload.get("history", [])], dtype=float
        )
        seed = int(payload["seed"])
    curve_metrics = None
    if method not in NON_CURVE_METHODS:
        if auc_curve.size != 200:
            raise ValueError(
                f"{path}: expected 200 round AUC values, found {auc_curve.size}"
            )
        curve_metrics = {
            "highest_auc": float(auc_curve.max()),
            "last10_auc": float(auc_curve[-10:].mean()),
        }
    return seed, final, curve_metrics


def mean_std(values):
    values = np.asarray(values, dtype=float) * 100.0
    return float(values.mean()), float(values.std(ddof=1))


def main():
    rows = []
    for method in METHODS:
        runs = [
            load_run(path, method)
            for path in sorted((ROOT / method).glob("seed*/*.json"))
        ]
        seeds = sorted(seed for seed, _, _ in runs)
        if seeds != [42, 43, 44, 45, 46]:
            raise SystemExit(f"Incomplete {method}: seeds={seeds}")
        bacc_mean, bacc_std = mean_std([run[1]["bacc"] for run in runs])
        auc_mean, auc_std = mean_std([run[1]["auc"] for run in runs])
        row = [method, 5, bacc_mean, bacc_std, auc_mean, auc_std]
        if method in NON_CURVE_METHODS:
            row.extend(("", "", "", ""))
        else:
            peak_mean, peak_std = mean_std(
                [run[2]["highest_auc"] for run in runs]
            )
            last_mean, last_std = mean_std(
                [run[2]["last10_auc"] for run in runs]
            )
            row.extend((peak_mean, peak_std, last_mean, last_std))
        rows.append(row)

    output = ROOT / "comparison_main2_table.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "method", "runs", "final_bacc_mean_pct", "final_bacc_std_pct",
            "final_auc_mean_pct", "final_auc_std_pct", "highest_auc_mean_pct",
            "highest_auc_std_pct", "last10_auc_mean_pct", "last10_auc_std_pct",
        ))
        for row in rows:
            writer.writerow([
                f"{value:.2f}" if isinstance(value, float) else value
                for value in row
            ])
    print(f"summary={output}; methods={len(rows)}; runs={len(rows) * 5}")


if __name__ == "__main__":
    main()
