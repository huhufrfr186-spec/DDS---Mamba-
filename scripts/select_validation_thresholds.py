"""Select and report DDS online-controller thresholds on LaSOT training validation only.

The script makes a deterministic one-coordinate-at-a-time sweep.  It never
opens LaSOT's public ``testing_set.txt`` and records every tried value so the
selected controller can be audited and used unchanged for public prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dds_mamba.assets import file_sha256, load_locked_yaml
from dds_mamba.data import lasot_sequences
from dds_mamba.geometry import iou
from dds_mamba.model_v1 import DDSV1
from dds_mamba.online import OnlineConfig
from dds_mamba.runtime import DDSTracker
from dds_mamba.splits import lasot_train_validation
from dds_mamba.variants import load_ablation_spec


def _cxcywh(xywh: np.ndarray) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, xywh)
    return x + width / 2.0, y + height / 2.0, width, height


@torch.no_grad()
def validation_success_auc(model: DDSV1, sequences: list, device: torch.device, overrides: dict[str, object]) -> float:
    """The same 21-threshold visible-frame AUC used for checkpoint selection."""
    overlaps: list[float] = []
    model.eval()
    for sequence in sequences:
        tracker = DDSTracker(model, device, overrides)
        tracker.initialize(sequence.rgb_frames[0], sequence.boxes_xywh[0])
        for frame, truth, visible in zip(sequence.rgb_frames[1:], sequence.boxes_xywh[1:], sequence.visible[1:]):
            prediction = tracker.update(frame)
            if visible:
                overlaps.append(0.0 if tracker.is_lost else iou(prediction, _cxcywh(truth)))
    if not overlaps:
        raise ValueError("validation split contains no visible frames")
    overlap = np.asarray(overlaps, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    return float(np.mean([(overlap >= threshold).mean() for threshold in thresholds]))


def _ordered_values(values: list[object], baseline: float) -> list[float]:
    numeric = [float(value) for value in values]
    if baseline not in numeric:
        raise ValueError(f"the threshold grid must include the base-manifest value {baseline}")
    # Baseline wins an exact tie.  The remaining order is numeric, which gives
    # the documented lower-value preference for a non-baseline tie.
    return [baseline] + [value for value in sorted(numeric) if value != baseline]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grid", type=Path, default=Path("configs/threshold_grid_v1.yaml"))
    parser.add_argument("--ablation-manifest", type=Path, default=Path("manifests/dds_mamba_v1_controls.yaml"))
    parser.add_argument("--variant", help="defaults to the variant stored in the checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, help="debug-only cap; omitted for the canonical full validation split")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    grid = load_locked_yaml(args.grid)
    manifest_sha = file_sha256(cfg["manifest"])
    if grid.get("base_manifest_sha256") != manifest_sha:
        raise RuntimeError("threshold grid is not bound to --config manifest")
    if grid.get("selection", {}).get("public_test_access") != "forbidden":
        raise ValueError("the canonical threshold grid must forbid public-test access")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_variant = checkpoint.get("variant")
    variant_name = args.variant or checkpoint_variant
    if variant_name is None:
        raise ValueError("checkpoint lacks variant metadata; pass --variant")
    if args.variant is not None and checkpoint_variant is not None and args.variant != checkpoint_variant:
        raise ValueError("--variant does not match checkpoint metadata")
    variant = load_ablation_spec(cfg["manifest"], variant_name, args.ablation_manifest)
    if checkpoint.get("manifest_sha256") not in (None, manifest_sha):
        raise RuntimeError("checkpoint base manifest does not match config manifest")
    if checkpoint.get("ablation_manifest_sha256") not in (None, variant.manifest_sha256):
        raise RuntimeError("checkpoint ablation manifest does not match --ablation-manifest")

    names = [line.strip() for line in (Path(cfg["lasot_root"]) / "training_set.txt").read_text().splitlines() if line.strip()]
    _, validation_names = lasot_train_validation(names)
    if args.max_sequences is not None:
        if args.max_sequences < 1:
            raise ValueError("--max-sequences must be positive")
        validation_names = validation_names[: args.max_sequences]
    sequences = lasot_sequences(cfg["lasot_root"], validation_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDSV1(cfg["manifest"], cfg["asset_dir"], variant=variant.name, network_overrides=dict(variant.network)).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    working = dict(variant.online)
    baseline_config = OnlineConfig.from_manifest(model.manifest, working)
    trials: list[dict[str, object]] = []
    baseline_auc = validation_success_auc(model, sequences, device, working)
    trials.append({"parameter": "__baseline__", "value": None, "success_auc": baseline_auc})

    for parameter, candidates in grid["parameters"].items():
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"threshold grid {parameter!r} must be a non-empty list")
        baseline_value = float(getattr(baseline_config, parameter)) if parameter not in working else float(working[parameter])
        scored: list[tuple[float, float]] = []
        for value in _ordered_values(candidates, baseline_value):
            candidate_overrides = dict(working)
            candidate_overrides[parameter] = value
            score = validation_success_auc(model, sequences, device, candidate_overrides)
            trials.append({"parameter": parameter, "value": value, "success_auc": score})
            scored.append((value, score))
        best_score = max(score for _, score in scored)
        tied = [value for value, score in scored if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
        selected = baseline_value if baseline_value in tied else min(tied)
        working[parameter] = selected

    report = {
        "schema_version": 1,
        "protocol": "LaSOT official training-set validation only; deterministic coordinate sweep; no public-test access",
        "selection_objective": "success_auc_21_thresholds",
        "manifest_sha256": manifest_sha,
        "ablation_manifest_sha256": variant.manifest_sha256,
        "threshold_grid_sha256": file_sha256(args.grid),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "variant": variant.name,
        "validation_split_id": "dds-mamba-v1-lasot-trainval-20260712",
        "validation_sequence_count": len(sequences),
        "debug_max_sequences": args.max_sequences,
        "base_success_auc": baseline_auc,
        "selected_online_overrides": working,
        "trials": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
