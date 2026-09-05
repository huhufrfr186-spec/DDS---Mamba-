"""Create a deterministic long-term tracking stress-test data set.

The set contains trajectories, target/distractor embeddings, explicit
occlusion intervals, and detector confidence. It is a *sanity benchmark*,
not a replacement for LaSOT or UAV benchmark data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("data/synthetic_longterm_v1")); p.add_argument("--sequences", type=int, default=24); p.add_argument("--frames", type=int, default=180); p.add_argument("--seed", type=int, default=20260711)
    a = p.parse_args(); rng = np.random.default_rng(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    meta = {"name": "DDS-Synthetic-LongTerm-v1", "seed": a.seed, "sequences": a.sequences, "frames": a.frames, "width": 640, "height": 480,
            "events": "two occlusions per sequence; first is distractor-only, second is target reappearance with appearance drift"}
    for seq in range(a.sequences):
        boxes = np.empty((a.frames, 4), np.float64); emb = np.empty((a.frames, 2), np.float64)
        visible = np.ones(a.frames, np.bool_); distractor = np.zeros(a.frames, np.bool_)
        centre = rng.uniform([150, 120], [490, 360]); velocity = rng.uniform([-3, -2], [3, 2]); size = rng.uniform([32, 28], [70, 62])
        phase = rng.uniform(-.2, .2); angle_rate = rng.uniform(.010, .015)
        first = 42 + (seq % 8); second = 112 + (seq % 10)
        intervals = [(first, first + 9), (second, second + 12)]
        for t in range(a.frames):
            velocity += rng.normal(0, .12, 2); velocity = np.clip(velocity, -4, 4); centre += velocity
            for axis, limit in enumerate((640, 480)):
                if centre[axis] < 50 or centre[axis] > limit - 50: velocity[axis] *= -1; centre[axis] = np.clip(centre[axis], 50, limit - 50)
            boxes[t] = (*centre, *size); theta = phase + angle_rate * t; emb[t] = (np.cos(theta), np.sin(theta))
            for lo, hi in intervals:
                if lo <= t < hi: visible[t] = False
            # The first four lost frames contain a plausible local distractor.
            if first <= t < first + 4 or second <= t < second + 4: distractor[t] = True
        np.savez_compressed(a.out / f"seq_{seq:03d}.npz", boxes=boxes, embeddings=emb, visible=visible, distractor=distractor)
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__": main()
