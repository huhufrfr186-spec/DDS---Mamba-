"""Launch every immutable controller/model ablation with the three locked seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dds_mamba.assets import load_locked_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    parser.add_argument("--ablation-manifest", type=Path, default=Path("manifests/dds_mamba_v1_ablations.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--variants", nargs="*", help="defaults to every locked variant")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    document = load_locked_yaml(args.ablation_manifest)
    variants = args.variants or list(document["variants"])
    unknown = set(variants) - set(document["variants"])
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    for variant in variants:
        command = [
            sys.executable,
            "scripts/run_three_seeds.py",
            "--config", str(args.config),
            "--variant", variant,
            "--ablation-manifest", str(args.ablation_manifest),
            "--output-root", str(args.output_root),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
