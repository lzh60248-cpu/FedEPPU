"""Merge split Section 4.5 timing jobs and build the paper tables.

The training array may be resubmitted in chunks, so results can live below
several Slurm job directories.  This utility selects the newest JSON for each
dataset/method/seed combination without requiring a job ID from the user.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


DATASETS = ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")
METHODS = ("fedavg_nnpu", "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu")
SEEDS = (42, 43, 44, 45, 46)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("communication_time_outputs"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("communication_time_artifacts"))
    return parser.parse_args()


def numeric_job_id(path: Path) -> int:
    match = re.fullmatch(r"\d+", path.name)
    return int(path.name) if match else -1


def candidates(root: Path, dataset: str, method: str, seed: int):
    pattern = f"*/{dataset}/{method}/seed{seed}/*.json"
    return sorted(root.glob(pattern), key=lambda item: numeric_job_id(item.parents[3]))


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Missing timing-result root: {args.input_root}")
    output = args.output_root
    merged = output / "merged_results"
    output.mkdir(parents=True, exist_ok=True)
    manifest, missing = [], []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                found = candidates(args.input_root, dataset, method, seed)
                if not found:
                    missing.append(f"{dataset}/{method}/seed{seed}")
                    continue
                source = found[-1]
                target = merged / dataset / method / f"seed{seed}" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                manifest.append({
                    "dataset": dataset, "method": method, "seed": seed,
                    "source": str(source), "selected_job": source.parents[3].name,
                    "merged": str(target),
                })

    with (output / "communication_time_merge_manifest.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "dataset", "method", "seed", "source", "selected_job", "merged"))
        writer.writeheader()
        writer.writerows(manifest)
    (output / "communication_time_missing_runs.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""), encoding="utf-8"
    )

    print(f"merged_results={len(manifest)}/120")
    print(f"merge_manifest={output / 'communication_time_merge_manifest.csv'}")
    if missing:
        print("missing_runs=" + ",".join(missing))
        print(f"missing_report={output / 'communication_time_missing_runs.txt'}")
        raise SystemExit(2)

    subprocess.run([
        sys.executable, "summarize_communication_time.py",
        "--input_dir", str(merged), "--output_dir", str(output),
    ], check=True)
    print(f"artifacts={output.resolve()}")


if __name__ == "__main__":
    main()
