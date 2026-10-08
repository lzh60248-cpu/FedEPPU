"""Create Table 7 statistics from Main2 inference-ablation artifacts."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("main2_inference_ablation_outputs")
VARIANTS = ("full", "last_raw", "zero_threshold", "last_raw_zero")


def main():
    grouped = {name: [] for name in VARIANTS}
    for path in ROOT.glob("*/seed*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped[payload["metadata"]["variant"]].append(payload["history"])
    output = ROOT / "main2_table7_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "runs", "bacc_mean_pct", "bacc_std_pct",
                         "auc_mean_pct", "auc_std_pct", "late_bacc_std_mean_pp",
                         "bacc_peak_to_final_drop_mean_pp"))
        for variant in VARIANTS:
            histories = grouped[variant]
            if len(histories) != 5:
                raise SystemExit(
                    f"Incomplete variant {variant}: expected 5 runs, found {len(histories)}"
                )
            selected_bacc = 100.0 * np.asarray([h["selected"]["bacc"] for h in histories])
            selected_auc = 100.0 * np.asarray([h["selected"]["auc"] for h in histories])
            late_std = 100.0 * np.asarray([np.std(h["bacc"][-20:], ddof=1) for h in histories])
            peak_drop = np.asarray([
                100.0 * (max(h["bacc"]) - h["bacc"][-1]) for h in histories
            ])
            writer.writerow((
                variant, len(histories), selected_bacc.mean(),
                selected_bacc.std(ddof=1) if len(histories) > 1 else 0.0,
                selected_auc.mean(),
                selected_auc.std(ddof=1) if len(histories) > 1 else 0.0,
                late_std.mean(), peak_drop.mean(),
            ))
    print(f"summary={output}; runs={sum(map(len, grouped.values()))}")


if __name__ == "__main__":
    main()
