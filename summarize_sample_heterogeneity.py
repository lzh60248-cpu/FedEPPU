import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("sample_heterogeneity_outputs")
records = [json.loads(path.read_text(encoding="utf-8"))
           for path in ROOT.rglob("*.json")]
if not records:
    raise SystemExit(f"No result JSON files found under {ROOT}")
groups = defaultdict(list)
for record in records:
    groups[(record["size_config"], record["aggregation"])].append(record)

output = ROOT / "sample_heterogeneity_summary.csv"
with output.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(("size_config", "aggregation", "runs",
                     "bacc_mean", "bacc_std", "auc_mean", "auc_std"))
    for key in sorted(groups):
        group = groups[key]
        bacc = np.asarray([r["metrics"]["bacc"] for r in group], float)
        auc = np.asarray([r["metrics"]["auc"] for r in group], float)
        ddof = 1 if len(group) > 1 else 0
        writer.writerow((*key, len(group), bacc.mean(), bacc.std(ddof=ddof),
                         auc.mean(), auc.std(ddof=ddof)))
print(output.resolve())
