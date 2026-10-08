"""Build the Section 4.2 four-dataset, five-seed comparison table."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DATASETS = ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")
METHODS = (
    "centralized_pn", "centralized_nnpu", "local_nnpu", "fedavg_nnpu",
    "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu",
)
SEEDS = (42, 43, 44, 45, 46)
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
    "fedpu": "FedPU (binary adaptation)",
    "mrfl_nnpu": "MRFL-nnPU adaptation",
    "fedeppu": "FedEPPU (Main3)",
}
METRICS = ("bacc", "acc", "auc", "f1", "max_auc", "last20_auc")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def result_path(root, dataset, method, seed):
    folder = root / dataset / method / f"seed{seed}"
    if method == "fedeppu":
        return folder / f"fedeppu_{dataset}_seed{seed}.json"
    return folder / f"{method}_{dataset}_seed{seed}.json"


def load_result(path, method):
    if not path.is_file():
        raise FileNotFoundError(f"Missing Section 4.2 result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method == "fedeppu":
        history_dict = payload["history"]
        selected = history_dict["selected"]
        history = [
            {key: float(history_dict[key][index]) for key in ("bacc", "acc", "auc", "f1")}
            for index in range(len(history_dict["auc"]))
        ]
        final = {key: float(selected[key]) for key in ("bacc", "acc", "auc", "f1")}
    else:
        history = payload.get("history", [])
        final = {key: float(payload["metrics"][key]) for key in ("bacc", "acc", "auc", "f1")}
    if len(history) != 200:
        raise ValueError(f"Expected 200 history rounds in {path}, found {len(history)}")
    auc = np.asarray([float(row["auc"]) for row in history], dtype=np.float64)
    if not np.isfinite(auc).all():
        raise ValueError(f"Non-finite AUC history in {path}")
    final["max_auc"] = float(auc.max())
    final["last20_auc"] = float(auc[-20:].mean())
    return final


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64) * 100.0
    return float(array.mean()), float(array.std(ddof=1))


def display(mean, sd):
    return f"{mean:.2f} ± {sd:.2f}"


def display_latex(mean, sd):
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    summary_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            values = {key: [] for key in METRICS}
            for seed in SEEDS:
                result = load_result(result_path(args.input_dir, dataset, method, seed), method)
                seed_row = {"dataset": dataset, "method": method, "seed": seed}
                for key in METRICS:
                    values[key].append(result[key])
                    seed_row[key] = result[key] * 100.0
                seed_rows.append(seed_row)
            row = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "method": method,
                "method_label": METHOD_LABELS[method],
            }
            for key in METRICS:
                row[f"{key}_mean"], row[f"{key}_sd"] = mean_sd(values[key])
            summary_rows.append(row)

    with (args.output_dir / "section42_seed_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (args.output_dir / "section42_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    header = "| Dataset | Method | BAcc (%) | ACC (%) | AUC (%) | F1 (%) | Max AUC (%) | Last-20 AUC (%) |"
    separator = "|---|---|---:|---:|---:|---:|---:|---:|"
    markdown = [header, separator]
    for row in summary_rows:
        cells = [row["dataset_label"], row["method_label"]]
        cells += [display(row[f"{key}_mean"], row[f"{key}_sd"]) for key in METRICS]
        markdown.append("| " + " | ".join(cells) + " |")
    (args.output_dir / "section42_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Comparison results on four datasets over five random seeds.}",
        r"\label{tab:section42_comparison}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccccc}",
        r"\toprule", r"Dataset & Method & BAcc & ACC & AUC & F1 & Max AUC & Last-20 AUC \\",
        r"\midrule",
    ]
    for index, row in enumerate(summary_rows):
        dataset = row["dataset_label"].replace("_", r"\_")
        method = row["method_label"].replace("_", r"\_")
        metrics = [display_latex(row[f"{key}_mean"], row[f"{key}_sd"]) for key in METRICS]
        latex.append(" & ".join([dataset, method] + metrics) + r" \\")
        if (index + 1) % len(METHODS) == 0 and index + 1 < len(summary_rows):
            latex.append(r"\midrule")
    latex += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    (args.output_dir / "section42_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(f"validated_results={len(seed_rows)}")
    print(f"summary_rows={len(summary_rows)}")
    print(f"table={args.output_dir / 'section42_table.md'}")


if __name__ == "__main__":
    main()
