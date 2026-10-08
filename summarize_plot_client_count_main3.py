"""Validate, summarize, and plot the Main3 client-count ablation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLIENT_COUNTS = (3, 6, 9, 12)
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("acc", "auc", "f1")
LABELS = {"acc": "ACC", "auc": "AUC", "f1": "F1"}
COLORS = {"acc": "#0072B2", "auc": "#D55E00", "f1": "#009E73"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Job directory containing clients3/, clients6/, clients9/, clients12/.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if values.size != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} seeds, got {values.size}.")
    return float(values.mean()), float(values.std(ddof=1))


def finite_curve(history, name):
    curve = np.asarray(history[name], dtype=float)
    if curve.size != 200:
        raise ValueError(f"{name} must contain 200 rounds, got {curve.size}.")
    if not np.isfinite(curve).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return curve * 100.0


def mean_actual_steps(history, n_clients):
    diagnostics = history.get("client_diag", [])
    if len(diagnostics) != 200:
        raise ValueError("client_diag must contain all 200 rounds.")
    if any(len(round_diag) != n_clients for round_diag in diagnostics):
        raise ValueError("client_diag does not match the requested client count.")
    values = [
        float(client["actual_steps_per_round"])
        for round_diag in diagnostics
        for client in round_diag
    ]
    return float(np.mean(values))


def load_run(path, expected_clients, expected_seed):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    history = payload["history"]
    args = metadata["args"]
    if int(args["n_clients"]) != expected_clients:
        raise ValueError(f"Wrong client count in {path}.")
    if int(args["seed"]) != expected_seed:
        raise ValueError(f"Wrong seed in {path}.")
    if int(args["rounds"]) != 200:
        raise ValueError(f"Expected 200 rounds in {path}.")
    if len(metadata["client_priors"]) != expected_clients:
        raise ValueError(f"Wrong prior count in {path}.")
    if len(metadata["client_sample_counts"]) != expected_clients:
        raise ValueError(f"Wrong sample-count vector in {path}.")
    selected = history.get("selected")
    if selected is None:
        raise ValueError(f"Selected checkpoint is missing in {path}.")
    curves = {name: finite_curve(history, name) for name in METRICS}
    return payload, selected, curves


def is_complete_root(root):
    return all(
        len(list((root / f"clients{n}" / f"seed{seed}").glob("*.json"))) == 1
        for n in CLIENT_COUNTS for seed in SEEDS
    )


def resolve_experiment_root(input_dir):
    """Select the fixed layout, or the newest complete legacy job directory."""
    if is_complete_root(input_dir):
        return input_dir
    candidates = [path for path in input_dir.iterdir()
                  if path.is_dir() and is_complete_root(path)]
    if not candidates:
        raise SystemExit(
            "No complete 20-run client-count experiment was found under "
            f"{input_dir}."
        )
    return max(
        candidates,
        key=lambda root: max(path.stat().st_mtime for path in root.rglob("*.json")),
    )


def main():
    args = parse_args()
    output = args.output_dir or args.input_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    experiment_root = resolve_experiment_root(args.input_dir)
    print(f"selected_experiment_root={experiment_root}")

    records = []
    missing = []
    for n_clients in CLIENT_COUNTS:
        for seed in SEEDS:
            relative = Path(f"clients{n_clients}") / f"seed{seed}"
            files = sorted((experiment_root / relative).glob("*.json"))
            if len(files) != 1:
                missing.append(
                    f"clients{n_clients}/seed{seed}: found {len(files)} JSON files"
                )
                continue
            payload, selected, curves = load_run(files[0], n_clients, seed)
            metadata = payload["metadata"]
            history = payload["history"]
            priors = np.asarray(metadata["client_priors"], dtype=float)
            counts = np.asarray(metadata["client_sample_counts"], dtype=int)
            row = {
                "n_clients": n_clients,
                "seed": seed,
                "selected_round": int(selected["round"]),
                "prior_min": float(priors.min()),
                "prior_max": float(priors.max()),
                "sample_count_min": int(counts.min()),
                "sample_count_max": int(counts.max()),
                "total_client_samples": int(counts.sum()),
                "mean_actual_steps_per_client_round": mean_actual_steps(
                    history, n_clients
                ),
            }
            for metric, curve in curves.items():
                row[f"selected_{metric}"] = float(selected[metric]) * 100.0
                row[f"last20_{metric}_mean"] = float(curve[-20:].mean())
                row[f"last20_{metric}_sd"] = float(curve[-20:].std(ddof=1))
                row[f"peak_to_final_{metric}_drop"] = float(
                    curve.max() - curve[-1]
                )
            records.append(row)

    if missing:
        raise SystemExit("Incomplete experiment:\n" + "\n".join(missing))
    if len(records) != len(CLIENT_COUNTS) * len(SEEDS):
        raise RuntimeError("The experiment must contain exactly 20 valid runs.")

    per_seed_path = output / "client_count_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    summary_fields = (
        "selected_acc", "selected_auc", "selected_f1",
        "last20_auc_mean", "last20_auc_sd", "peak_to_final_auc_drop",
        "prior_min", "prior_max", "sample_count_min", "sample_count_max",
        "mean_actual_steps_per_client_round",
    )
    for n_clients in CLIENT_COUNTS:
        group = [row for row in records if row["n_clients"] == n_clients]
        summary = {"n_clients": n_clients}
        for field in summary_fields:
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(
                [row[field] for row in group]
            )
        summary["client_updates_per_round"] = n_clients
        summary["nominal_optimizer_steps_per_round"] = (
            n_clients * 6 * 15
        )
        summaries.append(summary)

    summary_path = output / "client_count_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    table = [
        "| Clients | ACC (%) | AUC (%) | F1 (%) | Last-20 AUC SD | Prior range | Sample-count range | Nominal local steps/round |",
        "|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in summaries:
        table.append("| " + " | ".join([
            str(row["n_clients"]),
            f'{row["selected_acc_mean"]:.2f} ± {row["selected_acc_sd"]:.2f}',
            f'{row["selected_auc_mean"]:.2f} ± {row["selected_auc_sd"]:.2f}',
            f'{row["selected_f1_mean"]:.2f} ± {row["selected_f1_sd"]:.2f}',
            f'{row["last20_auc_sd_mean"]:.3f}',
            f'[{row["prior_min_mean"]:.4f}, {row["prior_max_mean"]:.4f}]',
            f'[{row["sample_count_min_mean"]:.0f}, {row["sample_count_max_mean"]:.0f}]',
            str(row["nominal_optimizer_steps_per_round"]),
        ]) + " |")
    table_path = output / "client_count_table.md"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")

    x = np.asarray(CLIENT_COUNTS, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for metric in METRICS:
        axes[0].errorbar(
            x,
            [row[f"selected_{metric}_mean"] for row in summaries],
            yerr=[row[f"selected_{metric}_sd"] for row in summaries],
            marker="o", linewidth=2.0, capsize=3,
            color=COLORS[metric], label=LABELS[metric],
        )
    axes[0].set_ylabel("Selected-checkpoint performance (%)")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        x,
        [row["last20_auc_sd_mean"] for row in summaries],
        yerr=[row["last20_auc_sd_sd"] for row in summaries],
        marker="s", linewidth=2.2, capsize=3, color="#CC5500",
    )
    axes[1].set_ylabel("Last-20-round AUC SD (points)")
    for axis, title in zip(axes, ("Performance", "Late-round fluctuation")):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Number of clients")
        axis.set_xticks(CLIENT_COUNTS)
        axis.grid(True, alpha=0.25)
    figure_png = output / "client_count_sensitivity.png"
    figure_pdf = output / "client_count_sensitivity.pdf"
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"validated_results={len(records)}")
    print(f"per_seed={per_seed_path}")
    print(f"summary={summary_path}")
    print(f"table={table_path}")
    print(f"figure_png={figure_png}")
    print(f"figure_pdf={figure_pdf}")


if __name__ == "__main__":
    main()
