"""Fetch (if needed) and verify every immutable frozen asset before a run."""
from __future__ import annotations

import argparse
from pathlib import Path

from dds_mamba.assets import asset, obtain_verified, load_asset_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/dds_mamba_v1.yaml"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    args = parser.parse_args()
    manifest = load_asset_lock(args.manifest)
    for name in ("template_search_encoder", "identity_encoder"):
        checkpoint = obtain_verified(asset(manifest, name), args.assets)
        print(f"verified {name}: {checkpoint}")


if __name__ == "__main__":
    main()
