"""Summarize extreme-prior ranking diagnostics from five Main3 runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44, 45, 46)
FIELDS = (
    "classification_loss_mean", "ranking_loss_mean",
    "classification_grad_norm_mean", "ranking_grad_norm_mean",
    "joint_grad_norm_before_clip_mean", "clip_active_fraction",
    "ranking_raw_risk_mean", "ranking_raw_risk_negative_fraction",
    "ranking_abs_correction_fraction", "nnpu_correction_fraction",
)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for seed in SEEDS:
        files = sorted((args.input_dir / f"seed{seed}").glob("*.json"))
        if len(files) != 1:
            missing.append(f"seed{seed}: found {len(files)} JSON files")
            continue
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        history = payload["history"]
        if len(history.get("auc", [])) != 200 or len(history.get("client_diag", [])) != 200:
            missing.append(f"seed{seed}: incomplete 200-round history")
            continue
        priors = payload["metadata"]["client_priors"]
        for round_index, round_diags in enumerate(history["client_diag"], start=1):
            for client_index, diag in enumerate(round_diags):
                row = {
                    "seed": seed,
                    "round": round_index,
                    "client": client_index + 1,
                    "prior": float(priors[client_index]),
                    "expected_negative_count": float(diag["expected_negative_count"]),
                    "ranking_support": float(diag["ranking_support_mean"]),
                    "global_auc": float(history["auc"][round_index - 1]) * 100.0,
                    "global_bacc": float(history["bacc"][round_index - 1]) * 100.0,
                }
                for field in FIELDS:
                    row[field] = float(diag[field])
                rows.append(row)
    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))

    detail_path = output / "rankdiag_client_round.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    summaries = []
    keys = sorted({(row["seed"], row["client"]) for row in rows})
    for seed, client in keys:
        group = [r for r in rows if r["seed"] == seed and r["client"] == client]
        summary = {
            "seed": seed, "client": client, "prior": group[0]["prior"],
            "expected_negative_count": group[0]["expected_negative_count"],
            "ranking_support": group[0]["ranking_support"],
        }
        for field in (*FIELDS, "global_auc", "global_bacc"):
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(
                [row[field] for row in group]
            )
            summary[f"{field}_max"] = max(row[field] for row in group)
        summaries.append(summary)

    summary_path = output / "rankdiag_client_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)

    lines = [
        "| Seed | Client | Prior | Expected negatives | q | Cls grad | Rank grad | Joint pre-clip | Clip rate | Negative rank-risk rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f'| {row["seed"]} | {row["client"]} | {row["prior"]:.6f} '
            f'| {row["expected_negative_count"]:.3f} | {row["ranking_support"]:.3f} '
            f'| {row["classification_grad_norm_mean_mean"]:.3f} '
            f'| {row["ranking_grad_norm_mean_mean"]:.3f} '
            f'| {row["joint_grad_norm_before_clip_mean_mean"]:.3f} '
            f'| {row["clip_active_fraction_mean"]:.3f} '
            f'| {row["ranking_raw_risk_negative_fraction_mean"]:.3f} |'
        )
    table_path = output / "rankdiag_table.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Detailed diagnostics -> {detail_path}")
    print(f"Per-client summary -> {summary_path}")
    print(f"Readable table -> {table_path}")


if __name__ == "__main__":
    main()
