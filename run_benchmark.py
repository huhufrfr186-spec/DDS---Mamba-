"""Run one manifest-locked benchmark protocol and write evaluator-ready rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dds_mamba.assets import file_sha256
from dds_mamba.datasets import ADAPTERS
from dds_mamba.model_v1 import DDSV1
from dds_mamba.runtime import DDSTracker
from dds_mamba.variants import load_ablation_spec, load_selected_thresholds


OUTPUT_DIR = {
    "lasot": Path("LaSOT") / "tracking_results" / "DDS-Mamba-v1",
    "anti_uav300_rgb": Path("Anti-UAV300-RGB") / "results" / "DDS-Mamba-v1",
    "webuav3m": Path("WebUAV-3M") / "results" / "DDS-Mamba-v1",
    "vtuav_rgb_longterm": Path("VTUAV") / "BB_results_RGB" / "DDS-Mamba-v1",
}


def xywh(box: tuple[float, float, float, float]) -> list[float]:
    return [box[0] - box[2] / 2.0, box[1] - box[3] / 2.0, box[2], box[3]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/dds_mamba_v1.yaml"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", choices=ADAPTERS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="Test", help="Anti-UAV300/WebUAV-3M split directory; ignored by LaSOT/VTUAV")
    parser.add_argument("--split-file", type=Path, help="Official VTUAV-V long-term list, if not embedded under root/test_LT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", help="required only for legacy checkpoints; otherwise read from checkpoint metadata")
    parser.add_argument("--ablation-manifest", type=Path, default=Path("manifests/dds_mamba_v1_ablations.yaml"))
    parser.add_argument("--threshold-selection", type=Path, help="validation-only threshold selection JSON produced by select_validation_thresholds.py")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_variant = checkpoint.get("variant")
    variant_name = args.variant or checkpoint_variant
    if variant_name is None:
        raise ValueError("checkpoint lacks variant metadata; pass --variant explicitly")
    if checkpoint_variant is not None and args.variant is not None and args.variant != checkpoint_variant:
        raise ValueError("--variant does not match the checkpoint metadata")
    variant = load_ablation_spec(args.manifest, variant_name, args.ablation_manifest)
    if checkpoint.get("manifest_sha256") not in (None, file_sha256(args.manifest)):
        raise RuntimeError("checkpoint base-manifest SHA-256 does not match --manifest")
    if checkpoint.get("ablation_manifest_sha256") not in (None, variant.manifest_sha256):
        raise RuntimeError("checkpoint ablation-manifest SHA-256 does not match --ablation-manifest")
    online_overrides = dict(variant.online)
    if args.threshold_selection is not None:
        online_overrides.update(
            load_selected_thresholds(
                args.threshold_selection,
                base_manifest=args.manifest,
                ablation_manifest_sha256=variant.manifest_sha256,
                checkpoint=args.checkpoint,
                variant=variant.name,
            )
        )
    model = DDSV1(args.manifest, args.assets, variant=variant.name, network_overrides=dict(variant.network))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    loader = ADAPTERS[args.benchmark]
    if args.benchmark == "lasot":
        sequences = loader(args.root)
    elif args.benchmark == "vtuav_rgb_longterm":
        sequences = loader(args.root, args.split_file)
    else:
        sequences = loader(args.root, args.split)
    destination = args.out / OUTPUT_DIR[args.benchmark]
    destination.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        # Standard OPE initialization is the only label access during inference.
        if not bool(sequence.visible[0]) or sequence.boxes_xywh[0, 2] <= 0 or sequence.boxes_xywh[0, 3] <= 0:
            raise ValueError(f"{sequence.name}: official protocol requires a visible first-frame initialization")
        tracker = DDSTracker(model, device, online_overrides)
        tracker.initialize(sequence.rgb_frames[0], sequence.boxes_xywh[0])
        predictions: list[list[float]] = [sequence.boxes_xywh[0].astype(float).tolist()]
        for image_path in sequence.rgb_frames[1:]:
            box = tracker.update(image_path)
            # Absence is a prediction from the state machine, never sequence.visible.
            predictions.append([0.0, 0.0, 0.0, 0.0] if tracker.is_lost else xywh(box))
        np.savetxt(destination / f"{sequence.name}.txt", np.asarray(predictions), delimiter=",", fmt="%.6f")
    provenance = {
        "benchmark": args.benchmark,
        "dataset_root": str(args.root.resolve()),
        "split": args.split,
        "split_file": None if args.split_file is None else str(args.split_file.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "ablation_manifest_sha256": variant.manifest_sha256,
        "variant": variant.name,
        "variant_description": variant.description,
        "threshold_selection": None if args.threshold_selection is None else str(args.threshold_selection.resolve()),
        "threshold_selection_sha256": None if args.threshold_selection is None else file_sha256(args.threshold_selection),
        "online_overrides": online_overrides,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version,
        "sequence_count": len(sequences),
        "no_test_label_access_after_initialization": True,
    }
    (destination / "run_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
