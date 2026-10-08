"""Summarize the 30 controlled prior-heterogeneity runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONFIGS = ("low", "medium", "extreme")
METHODS = ("fedavg_nnpu", "fedeppu")
SEEDS = (42, 43, 44, 45, 46)
LABELS = {"fedavg_nnpu": "FedAvg-nnPU", "fedeppu": "FedEPPU"}
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
                if payload["priors"] != list({
                    "low": (0.4, 0.5, 0.6),
                    "medium": (0.2, 0.5, 0.8),
                    "extreme": (0.05, 0.5, 0.95),
                }[config]):
                    raise ValueError(f"Prior mismatch in {path}")
                runs.append(payload["metrics"])
            row = {"configuration": config, "method": LABELS[method]}
            for metric in METRICS:
                values = np.asarray([run[metric] for run in runs]) * 100.0
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_sd"] = values.std(ddof=1)
            rows.append(row)
    output = args.input_dir / "prior_heterogeneity_summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| Prior heterogeneity | Method | BAcc (%) | ACC (%) | AUC (%) | F1 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_sd']:.2f}"
            for metric in METRICS
        ]
        markdown.append("| " + " | ".join(
            [row["configuration"], row["method"]] + values
        ) + " |")
    table = args.input_dir / "prior_heterogeneity_table.md"
    table.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("validated_results=30")
    print(f"summary={output}")
    print(f"table={table}")


if __name__ == "__main__":
    main()
