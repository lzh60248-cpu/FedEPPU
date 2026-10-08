import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("rank_correction_outputs")
CORRECTIONS = ("absolute", "clamp")
SEEDS = tuple(range(42, 47))

rows = []
objects = {}
for correction in CORRECTIONS:
    for seed in SEEDS:
        files = list((ROOT / correction / f"seed{seed}").glob("*.json"))
        if len(files) != 1:
            raise SystemExit(
                f"expected_one_json correction={correction} seed={seed} found={len(files)}"
            )
        obj = json.loads(files[0].read_text(encoding="utf-8"))
        objects[(correction, seed)] = obj
        history = obj["history"]
        selected = history["selected"]
        late_auc = history["auc"][-20:]
        rows.append({
            "correction": correction,
            "seed": seed,
            "selected_round": selected["round"],
            "acc": selected["acc"],
            "f1": selected["f1"],
            "bacc": selected["bacc"],
            "auc": selected["auc"],
            "last20_auc_mean": mean(late_auc),
            "last20_auc_sd": stdev(late_auc),
            "peak_to_final_auc": max(history["auc"]) - history["auc"][-1],
        })

with open("rank_correction_per_seed.csv", "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = []
for correction in CORRECTIONS:
    subset = [row for row in rows if row["correction"] == correction]
    item = {"correction": correction}
    for key in (
        "acc", "f1", "bacc", "auc", "last20_auc_mean",
        "last20_auc_sd", "peak_to_final_auc",
    ):
        values = [100.0 * row[key] for row in subset]
        item[key] = f"{mean(values):.2f} ± {stdev(values):.2f}"
    summary.append(item)

with open("rank_correction_summary.csv", "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

diagnostics = []
for correction in CORRECTIONS:
    active_records = []
    all_records = []
    for seed in SEEDS:
        for round_diag in objects[(correction, seed)]["history"]["client_diag"]:
            for diag in round_diag:
                all_records.append(diag)
                if float(diag.get("ranking_active_fraction", 0.0)) > 0.0:
                    active_records.append(diag)
    diagnostics.append({
        "correction": correction,
        "all_client_round_records": len(all_records),
        "ranking_active_records": len(active_records),
        "raw_negative_fraction": mean(
            float(x["ranking_raw_risk_negative_fraction"]) for x in active_records
        ),
        "ranking_loss_mean": mean(float(x["ranking_loss_mean"]) for x in active_records),
        "effective_ranking_grad_norm": mean(
            float(x["effective_ranking_grad_norm_mean"]) for x in active_records
        ),
        "joint_nonfinite_fraction": mean(
            float(x["joint_grad_nonfinite_fraction"]) for x in active_records
        ),
        "clip_active_fraction": mean(
            float(x["clip_active_fraction"]) for x in active_records
        ),
    })

with open("rank_correction_diagnostics.csv", "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=diagnostics[0].keys())
    writer.writeheader()
    writer.writerows(diagnostics)

paired = []
for metric in ("acc", "f1", "bacc", "auc", "last20_auc_mean", "last20_auc_sd"):
    differences = []
    for seed in SEEDS:
        clamp = next(r for r in rows if r["correction"] == "clamp" and r["seed"] == seed)
        absolute = next(r for r in rows if r["correction"] == "absolute" and r["seed"] == seed)
        differences.append(100.0 * (clamp[metric] - absolute[metric]))
    effect = mean(differences)
    sd = stdev(differences)
    half_width = 2.776 * sd / math.sqrt(len(SEEDS))
    paired.append({
        "comparison": "clamp - absolute",
        "metric": metric,
        **{f"seed{seed}": value for seed, value in zip(SEEDS, differences)},
        "mean_effect_pp": effect,
        "sample_sd_pp": sd,
        "ci95_low_pp": effect - half_width,
        "ci95_high_pp": effect + half_width,
        "positive_seeds": sum(value > 0 for value in differences),
        "negative_seeds": sum(value < 0 for value in differences),
        "zero_seeds": sum(value == 0 for value in differences),
    })

with open("rank_correction_paired_effects.csv", "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=paired[0].keys())
    writer.writeheader()
    writer.writerows(paired)

print("completed_runs=10")
print("summary=rank_correction_summary.csv")
print("per_seed=rank_correction_per_seed.csv")
print("diagnostics=rank_correction_diagnostics.csv")
print("paired_effects=rank_correction_paired_effects.csv")
