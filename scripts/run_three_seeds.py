"""Launch the three manifest-locked training seeds for one DDS-Mamba variant."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from dds_mamba.assets import load_asset_lock
from dds_mamba.variants import load_ablation_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    parser.add_argument("--variant", default="full")
    parser.add_argument("--ablation-manifest", type=Path, default=Path("manifests/dds_mamba_v1_ablations.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest = load_asset_lock(config["manifest"])
    spec = load_ablation_spec(config["manifest"], args.variant, args.ablation_manifest)
    for seed in manifest["training"]["random_seeds"]:
        run_dir = args.output_root / f"{spec.name}_seed_{int(seed)}"
        run_config = dict(config)
        run_config["seed"] = int(seed)
        run_config["output_dir"] = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        materialized = run_dir / "train_config.yaml"
        materialized.write_text(yaml.safe_dump(run_config, sort_keys=True))
        command = [sys.executable, "train_v1.py", "--config", str(materialized), "--variant", spec.name, "--ablation-manifest", str(args.ablation_manifest)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
