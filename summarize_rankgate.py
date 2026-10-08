"""Summarize the matched original/continuous/hard ranking-gate experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


VARIANTS = ("original", "continuous", "hard")
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("acc", "f1", "bacc", "auc")
DIAGNOSTICS = (
    "classification_loss_mean", "ranking_loss_mean",
    "classification_grad_norm_mean", "ranking_grad_norm_mean",
    "effective_ranking_grad_norm_mean", "joint_grad_norm_before_clip_mean",
    "joint_grad_nonfinite_fraction", "clip_active_fraction",
    "ranking_raw_risk_negative_fraction",
)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--require_strict", action="store_true",
        help="Reject runs that were not launched with --strict_determinism.",
    )
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)

    performance, clients, missing = [], [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            files = sorted((args.input_dir / variant / f"seed{seed}").glob("*.json"))
            if len(files) != 1:
                missing.append(f"{variant}/seed{seed}: found {len(files)} JSON files")
                continue
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            strict = bool(payload.get("metadata", {}).get("args", {}).get(
                "strict_determinism", False
            ))
            if args.require_strict and not strict:
                missing.append(f"{variant}/seed{seed}: strict_determinism is false")
                continue
            history = payload["history"]
            if len(history.get("auc", [])) != 200 or len(history.get("client_diag", [])) != 200:
                missing.append(f"{variant}/seed{seed}: incomplete 200-round history")
                continue
            selected = history["selected"]
            row = {
                "variant": variant, "seed": seed,
                "strict_determinism": strict,
                "selected_round": int(selected["round"]),
            }
            for metric in METRICS:
                curve = np.asarray(history[metric], dtype=float) * 100.0
                row[metric] = float(selected[metric]) * 100.0
                row[f"last20_{metric}_sd"] = float(curve[-20:].std(ddof=1))
            performance.append(row)

            priors = payload["metadata"]["client_priors"]
            for client_index, prior in enumerate(priors):
                diags = [round_diags[client_index] for round_diags in history["client_diag"]]
                client_row = {
                    "variant": variant, "seed": seed, "client": client_index + 1,
                    "prior": float(prior),
                    "expected_negative_count": float(diags[0]["expected_negative_count"]),
                    "ranking_support": float(diags[0]["ranking_support_mean"]),
                    "local_ranking_scale": float(diags[0]["ranking_scale_mean"]),
                }
                for field in DIAGNOSTICS:
                    client_row[field] = float(np.mean([float(d[field]) for d in diags]))
                clients.append(client_row)
    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))

    perf_path = output / "rankgate_per_seed.csv"
    with perf_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=performance[0].keys())
        writer.writeheader(); writer.writerows(performance)
    client_path = output / "rankgate_client_diagnostics.csv"
    with client_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=clients[0].keys())
        writer.writeheader(); writer.writerows(clients)

    summaries = []
    for variant in VARIANTS:
        group = [row for row in performance if row["variant"] == variant]
        summary = {"variant": variant}
        for field in (*METRICS, "last20_auc_sd"):
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(
                [row[field] for row in group]
            )
        extreme = [
            row for row in clients
            if row["variant"] == variant and row["prior"] >= 0.99
        ]
        for field in (
            "ranking_loss_mean", "ranking_grad_norm_mean",
            "effective_ranking_grad_norm_mean", "joint_grad_nonfinite_fraction",
            "clip_active_fraction",
        ):
            summary[f"extreme_{field}_mean"] = float(np.mean([row[field] for row in extreme]))
        summaries.append(summary)

    summary_path = output / "rankgate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)

    lines = [
        "| Variant | ACC (%) | F1 (%) | BAcc (%) | AUC (%) | Last-20 AUC SD | Extreme effective rank grad | Extreme nonfinite rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f'| {row["variant"]} '
            f'| {row["acc_mean"]:.2f} ± {row["acc_sd"]:.2f} '
            f'| {row["f1_mean"]:.2f} ± {row["f1_sd"]:.2f} '
            f'| {row["bacc_mean"]:.2f} ± {row["bacc_sd"]:.2f} '
            f'| {row["auc_mean"]:.2f} ± {row["auc_sd"]:.2f} '
            f'| {row["last20_auc_sd_mean"]:.3f} '
            f'| {row["extreme_effective_ranking_grad_norm_mean_mean"]:.3f} '
            f'| {row["extreme_joint_grad_nonfinite_fraction_mean"]:.3f} |'
        )
    table_path = output / "rankgate_table.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Per-seed metrics -> {perf_path}")
    print(f"Client diagnostics -> {client_path}")
    print(f"Summary -> {summary_path}")
    print(f"Readable table -> {table_path}")


if __name__ == "__main__":
    main()
