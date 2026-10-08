"""Merge split Dirichlet comparison jobs and build the table and figure."""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


CONFIGS = ("alpha0p1", "alpha0p5", "alpha1p0", "alpha2p0")
METHODS = ("fedavg_nnpu", "fedeppu")
SEEDS = (42, 43, 44, 45, 46)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, default=Path("dirichlet_comparison_outputs"))
    parser.add_argument("--output_dir", type=Path, default=Path("dirichlet_comparison_artifacts"))
    return parser.parse_args()


def job_id(path, root):
    for part in path.relative_to(root).parts:
        if part.isdigit():
            return int(part)
    return -1


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"Missing input root: {args.input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = args.output_dir / "merged_results"
    merged.mkdir(parents=True, exist_ok=True)
    manifest, missing = [], []

    for config in CONFIGS:
        for method in METHODS:
            for seed in SEEDS:
                candidates = list((args.input_root).glob(
                    f"*/{config}/{method}/seed{seed}/*.json"
                ))
                if not candidates:
                    missing.append(f"{config}/{method}/seed{seed}")
                    continue
                selected = max(
                    candidates,
                    key=lambda path: (job_id(path, args.input_root), path.stat().st_mtime_ns),
                )
                destination = merged / config / method / f"seed{seed}" / selected.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(selected, destination)
                manifest.append({
                    "config": config,
                    "method": method,
                    "seed": seed,
                    "selected_job_id": job_id(selected, args.input_root),
                    "candidate_count": len(candidates),
                    "source": str(selected),
                })

    manifest_path = args.output_dir / "dirichlet_comparison_merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("config", "method", "seed", "selected_job_id", "candidate_count", "source"),
        )
        writer.writeheader()
        writer.writerows(manifest)
    if missing:
        raise SystemExit(
            f"Incomplete Dirichlet comparison: found {len(manifest)}/40 results.\n"
            + "\n".join(missing)
        )

    project_root = Path(__file__).resolve().parent
    subprocess.run([
        sys.executable, str(project_root / "summarize_plot_dirichlet_comparison_main3.py"),
        "--input_dir", str(merged), "--output_dir", str(args.output_dir),
    ], check=True)
    print("merged_unique_results=40")
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
