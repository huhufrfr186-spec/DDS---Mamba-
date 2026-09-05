"""Measure device-resident DDS-Mamba inference latency with a saved checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dds_mamba.assets import file_sha256
from dds_mamba.datasets import ADAPTERS
from dds_mamba.model_v1 import DDSV1
from dds_mamba.runtime import DDSTracker
from dds_mamba.variants import load_ablation_spec, load_selected_thresholds


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/dds_mamba_v1.yaml"))
    parser.add_argument("--ablation-manifest", type=Path, default=Path("manifests/dds_mamba_v1_ablations.yaml"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", choices=ADAPTERS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="Test")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-selection", type=Path, help="optional validation-only threshold selection JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("runtime protocol requires a CUDA device")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = load_ablation_spec(args.manifest, checkpoint.get("variant", "full"), args.ablation_manifest)
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
    device = torch.device("cuda")
    model = DDSV1(args.manifest, args.assets, variant=variant.name, network_overrides=dict(variant.network)).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = ADAPTERS[args.benchmark]
    if args.benchmark == "lasot":
        sequences = loader(args.root)
    elif args.benchmark == "vtuav_rgb_longterm":
        sequences = loader(args.root, args.split_file)
    else:
        sequences = loader(args.root, args.split)
    sequence = sequences[args.sequence_index]
    count = args.warmup + args.measure + 1
    if len(sequence.rgb_frames) < count:
        raise ValueError(f"{sequence.name} has {len(sequence.rgb_frames)} frames; need at least {count}")
    # Preload to GPU before timing; the protocol explicitly excludes disk I/O,
    # decode, and host-to-device transfer.
    from dds_mamba.data import image_tensor

    images = [image_tensor(path).unsqueeze(0).to(device) for path in sequence.rgb_frames[:count]]
    tracker = DDSTracker(model, device, online_overrides)
    tracker.initialize_image(images[0], sequence.boxes_xywh[0])
    for image in images[1 : args.warmup + 1]:
        tracker.update_image(image)
    torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    for image in images[args.warmup + 1 :]:
        _sync(device)
        start = time.perf_counter_ns()
        tracker.update_image(image)
        _sync(device)
        times.append((time.perf_counter_ns() - start) / 1e6)
    # PyTorch profiler reports FLOPs for the operators it recognizes.  The
    # resulting field deliberately retains that qualifier instead of silently
    # claiming a hand-derived MAC count for custom selective-SSM operations.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as profiler:
        tracker.update_image(images[-1])
    profiler_flops = int(sum(int(event.flops or 0) for event in profiler.key_averages()))
    report = {
        "protocol": "device-resident RGB update; excludes disk I/O, image decode, and host-to-device transfer",
        "benchmark": args.benchmark,
        "sequence": sequence.name,
        "warmup_frames": args.warmup,
        "measured_frames": len(times),
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.quantile(times, 0.90)),
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_count_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "torch_profiler_reported_flops_one_update": profiler_flops,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "variant": variant.name,
        "threshold_selection": None if args.threshold_selection is None else str(args.threshold_selection.resolve()),
        "threshold_selection_sha256": None if args.threshold_selection is None else file_sha256(args.threshold_selection),
        "online_overrides": online_overrides,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
