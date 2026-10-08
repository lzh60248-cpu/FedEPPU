"""Verify identity where all ranking-gate formulas reduce to scale one."""
import argparse
import json
from pathlib import Path


VARIANTS = ("original", "continuous", "hard")
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("acc", "auc", "f1", "bacc", "recall", "specificity", "threshold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    args = parser.parse_args()
    checked = 0
    for seed in SEEDS:
        payloads = {}
        all_unit_scale = True
        for variant in VARIANTS:
            files = sorted((args.input_dir / variant / f"seed{seed}").glob("*.json"))
            if len(files) != 1:
                raise SystemExit(f"{variant}/seed{seed}: expected one JSON")
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            if not payload["metadata"]["args"].get("strict_determinism", False):
                raise SystemExit(f"{variant}/seed{seed}: strict_determinism is false")
            payloads[variant] = payload
            scales = [
                float(client["ranking_scale_mean"])
                for round_diag in payload["history"]["client_diag"]
                for client in round_diag
            ]
            all_unit_scale = all_unit_scale and all(abs(x - 1.0) <= 1e-12 for x in scales)
        if not all_unit_scale:
            continue
        checked += 1
        reference = payloads["original"]["history"]
        for variant in ("continuous", "hard"):
            candidate = payloads[variant]["history"]
            for metric in METRICS:
                if reference[metric] != candidate[metric]:
                    raise SystemExit(
                        f"identity_check_failed seed={seed} variant={variant} metric={metric}"
                    )
        print(f"identity_check=ok seed={seed}; all local ranking scales equal one")
    if checked == 0:
        raise SystemExit("identity_check_not_applicable: no seed had unit scales throughout")
    print(f"strict_identity_checks_passed={checked}")


if __name__ == "__main__":
    main()
