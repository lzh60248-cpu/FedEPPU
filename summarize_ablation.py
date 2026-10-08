"""Create Table 6 statistics from completed ablation JSON artifacts."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("ablation_outputs")


def main():
    records = []
    for path in ROOT.glob("*/seed*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append((payload["metadata"]["variant"], payload["history"]["selected"]))
    if not records:
        raise SystemExit("No ablation JSON files found")
    output = ROOT / "table6_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "runs", "bacc_mean", "bacc_std",
                         "auc_mean", "auc_std"))
        for variant in ("full", "no_ranking", "no_complement", "no_support", "fedavg"):
            selected = [item for name, item in records if name == variant]
            bacc = np.asarray([item["bacc"] for item in selected], float)
            auc = np.asarray([item["auc"] for item in selected], float)
            writer.writerow((variant, len(selected), bacc.mean(),
                             bacc.std(ddof=1) if len(bacc) > 1 else 0.0,
                             auc.mean(), auc.std(ddof=1) if len(auc) > 1 else 0.0))
    print(f"summary={output}; files={len(records)}")


if __name__ == "__main__":
    main()
