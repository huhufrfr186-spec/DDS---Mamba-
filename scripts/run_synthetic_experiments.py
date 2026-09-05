"""Deterministic V1-controller regression experiment (no neural encoders)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dds_mamba.geometry import iou, roi_ratio
from dds_mamba.online import Candidate, DDSOnlineState, OnlineConfig


def _candidate(
    tracker: DDSOnlineState,
    predicted: tuple[float, float, float, float],
    gt: np.ndarray,
    embedding: np.ndarray,
    visible: bool,
    distractor: bool,
    rng: np.random.Generator,
) -> Candidate:
    """A fixed noisy detector used only for controller-level regression tests."""
    if distractor:
        box = (tracker.box[0] + rng.normal(8, 2), tracker.box[1] + rng.normal(6, 2), float(gt[2]), float(gt[3]))
        candidate_embedding, map_q, peak = -embedding, 0.95, 0.95
    elif visible:
        box = tuple(np.asarray(gt, dtype=np.float64) + rng.normal([0, 0, 0, 0], [3, 3, 1, 1]))
        candidate_embedding, map_q, peak = embedding, 0.95, 0.95
    else:
        box = (tracker.box[0] + rng.normal(12, 3), tracker.box[1] + rng.normal(-10, 3), float(gt[2]), float(gt[3]))
        candidate_embedding, map_q, peak = -embedding, 0.90, 0.90
    box = tuple(map(float, box))
    return Candidate(
        box,
        map_q,
        peak,
        iou(box, predicted),
        float(candidate_embedding[0]),
        candidate_embedding.copy(),
        np.asarray(box[:2], dtype=np.float32),
        candidate_embedding.copy(),
        np.asarray(box[:2], dtype=np.float32),
        roi_ratio(box, tracker.width, tracker.height),
        0,
        0.0,
    )


def _run_variant(files: list[Path], name: str, overrides: dict[str, object]) -> dict[str, float | int | str]:
    total_iou = total_frames = false_active = recovery_events = recovery_sum = 0
    for ordinal, path in enumerate(files):
        data = np.load(path)
        boxes, embeddings, visible, distractor = data["boxes"], data["embeddings"], data["visible"], data["distractor"]
        tracker = DDSOnlineState(
            tuple(map(float, boxes[0])),
            embeddings[0] if bool(overrides.get("use_identity", True)) else None,
            2,
            640,
            480,
            OnlineConfig(**overrides),
        )
        rng = np.random.default_rng(1_000 + ordinal)
        was_lost, began = False, None
        for frame in range(1, len(boxes)):
            predicted = tracker.predict()
            proposal = _candidate(
                tracker, predicted, boxes[frame], embeddings[frame], bool(visible[frame]), bool(distractor[frame]), rng
            )
            output = tracker.step([proposal], predicted)
            total_iou += iou(output, tuple(map(float, boxes[frame])))
            total_frames += 1
            false_active += int((not visible[frame]) and tracker.mode.value == "active")
            if not visible[frame] and tracker.mode.value == "lost" and not was_lost:
                was_lost, began = True, frame
            if was_lost and visible[frame] and tracker.mode.value == "active":
                recovery_events += 1
                recovery_sum += frame - int(began)
                was_lost = False
    return {
        "variant": name,
        "mean_iou": total_iou / total_frames,
        "false_active_frames": false_active,
        "recovery_events": recovery_events,
        "mean_recovery_delay_frames": recovery_sum / max(recovery_events, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_longterm_v1"))
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_longterm_v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data.glob("seq_*.npz"))
    if not files:
        raise FileNotFoundError(f"no sequences under {args.data}")
    rows = [
        _run_variant(files, "DDS-Mamba-v1 controller (full)", {}),
        _run_variant(files, "DDS-Mamba-v1 controller (no RFMB)", {"memory_capacity": 0, "enable_rfmb": False}),
        _run_variant(files, "DDS-Mamba-v1 controller (no identity gate)", {"use_identity": False}),
    ]
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "metrics.json").write_text(
        json.dumps({"protocol": "DDS-Synthetic-LongTerm-v1", "controller": "DDSOnlineState-v1", "metrics": rows}, indent=2) + "\n"
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
