"""Merge split Main3 loss-diagnostic runs and render Figure 15."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("main3_local_loss_outputs"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("main3_local_loss_artifacts"))
    return parser.parse_args()


def job_number(path: Path) -> int:
    return int(path.name) if re.fullmatch(r"\d+", path.name) else -1


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Missing loss-result root: {args.input_root}")
    output = args.output_root
    merged = output / "merged_results"
    selected = []
    for seed in SEEDS:
        candidates = sorted(
            args.input_root.glob(f"*/seed{seed}/main3_local_loss_cifar10_seed{seed}.json"),
            key=lambda path: job_number(path.parents[1]),
        )
        if not candidates:
            raise FileNotFoundError(f"Missing Main3 local-loss JSON for seed {seed}")
        source = candidates[-1]
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("metadata", {}).get("method") != "FedEPPU-Main3-LocalLossCurves":
            raise ValueError(f"Unexpected result protocol: {source}")
        target = merged / f"seed{seed}" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected.append((seed, source.parents[1].name, source))

    manifest = output / "loss_merge_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(
        f"seed{seed},job={job},source={source}" for seed, job, source in selected
    ) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, "plot_main3_local_losses.py",
        "--root", str(merged),
        "--output", str(output / "figure15_local_losses.pdf"),
    ], check=True)
    print("validated_results=5")
    print(f"artifacts={output.resolve()}")


if __name__ == "__main__":
    main()
