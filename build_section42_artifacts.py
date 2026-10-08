"""Merge multi-job Section 4.2 JSON results and build paper artifacts on CPU."""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


DATASETS = ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")
METHODS = (
    "centralized_pn", "centralized_nnpu", "local_nnpu", "fedavg_nnpu",
    "fedprox_nnpu", "scaffold_nnpu", "fedpu", "mrfl_nnpu", "fedeppu",
)
SEEDS = (42, 43, 44, 45, 46)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge results from multiple Slurm job directories, then summarize and plot."
    )
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def expected_name(dataset, method, seed):
    if method == "fedeppu":
        return f"fedeppu_{dataset}_seed{seed}.json"
    return f"{method}_{dataset}_seed{seed}.json"


def job_id(path, input_root):
    relative = path.relative_to(input_root)
    for part in relative.parts:
        if part.isdigit():
            return int(part)
    return -1


def candidates(input_root, dataset, method, seed):
    suffix = (dataset, method, f"seed{seed}", expected_name(dataset, method, seed))
    found = []
    for path in input_root.rglob(suffix[-1]):
        if tuple(path.parts[-4:]) == suffix:
            found.append(path)
    return found


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {args.input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = args.output_dir / "merged_results"
    merged.mkdir(parents=True, exist_ok=True)
    manifest, missing = [], []

    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                found = candidates(args.input_root, dataset, method, seed)
                if not found:
                    missing.append(f"{dataset}/{method}/seed{seed}")
                    continue
                # Prefer the highest numeric Slurm job ID. Modification time is
                # a deterministic fallback for layouts without numeric job folders.
                chosen = max(found, key=lambda p: (job_id(p, args.input_root), p.stat().st_mtime_ns))
                destination = merged / dataset / method / f"seed{seed}" / expected_name(
                    dataset, method, seed
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chosen, destination)
                manifest.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "selected_job_id": job_id(chosen, args.input_root),
                    "candidate_count": len(found),
                    "source": str(chosen),
                    "merged": str(destination),
                })

    manifest_path = args.output_dir / "section42_merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys() if manifest else (
            "dataset", "method", "seed", "selected_job_id", "candidate_count", "source", "merged"
        ))
        writer.writeheader(); writer.writerows(manifest)

    if missing:
        details = "\n".join(missing)
        raise SystemExit(
            f"Incomplete Section 4.2 results: found {len(manifest)}/180 unique runs.\n"
            f"Missing:\n{details}\nManifest: {manifest_path}"
        )
    if len(manifest) != 180:
        raise RuntimeError(f"Expected 180 merged results, found {len(manifest)}")

    project_root = Path(__file__).resolve().parent
    subprocess.run([
        sys.executable, str(project_root / "summarize_section42.py"),
        "--input_dir", str(merged), "--output_dir", str(args.output_dir),
    ], check=True)
    subprocess.run([
        sys.executable, str(project_root / "plot_section42_curves.py"),
        "--input_dir", str(merged), "--output_dir", str(args.output_dir),
    ], check=True)
    print("merged_unique_results=180")
    print(f"manifest={manifest_path}")
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
