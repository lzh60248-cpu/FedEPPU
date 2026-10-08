"""Summarize the 45 controlled sample-heterogeneity runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONFIGS = ("balanced", "moderate", "extreme")
METHODS = ("fedavg_nnpu", "fedeppu_no_complement", "fedeppu")
SEEDS = (42, 43, 44, 45, 46)
LABELS = {
    "fedavg_nnpu": "FedAvg-nnPU",
    "fedeppu_no_complement": "FedEPPU w/o Complementary Mass",
    "fedeppu": "FedEPPU",
}
METRICS = ("bacc", "acc", "auc", "f1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for config in CONFIGS:
        for method in METHODS:
            runs = []
            for seed in SEEDS:
                path = (args.input_dir / config / method / f"seed{seed}" /
                        f"{method}_{config}_seed{seed}.json")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing result: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                runs.append(payload["metrics"])
            row = {"sample_ratio": config, "method": LABELS[method]}
            for metric in METRICS:
                values = np.asarray([run[metric] for run in runs]) * 100.0
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_sd"] = values.std(ddof=1)
            rows.append(row)
    csv_path = args.input_dir / "sample_heterogeneity_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| Sample ratio | Method | BAcc (%) | ACC (%) | AUC (%) | F1 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_sd']:.2f}"
            for metric in METRICS
        ]
        markdown.append("| " + " | ".join(
            [row["sample_ratio"], row["method"]] + values
        ) + " |")
    table = args.input_dir / "sample_heterogeneity_table.md"
    table.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("validated_results=45")
    print(f"summary={csv_path}")
    print(f"table={table}")


if __name__ == "__main__":
    main()
