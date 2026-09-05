"""Strict adapters for the four manifest-locked public benchmark layouts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import Sequence, lasot_sequences


def _frames(path: Path) -> tuple[Path, ...]:
    return tuple(sorted([*path.glob("*.jpg"), *path.glob("*.png"), *path.glob("*.jpeg")]))


def _boxes(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1, 4)
    except ValueError:
        return np.loadtxt(path, dtype=np.float32).reshape(-1, 4)


def _names(path: Path) -> list[str]:
    text = path.read_text().strip()
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"split file must be a JSON list of sequence names: {path}")
        return value
    return [line.strip() for line in text.splitlines() if line.strip()]


def lasot_official_test(root: str | Path) -> list[Sequence]:
    """Protocol II only; the official archive ships ``testing_set.txt``."""
    root = Path(root)
    split = root / "testing_set.txt"
    if not split.exists():
        raise FileNotFoundError("LaSOT Protocol II requires the official root/testing_set.txt (280 sequences)")
    names = _names(split)
    if len(names) != 280:
        raise ValueError(f"expected 280 official LaSOT test names, found {len(names)} in {split}")
    records = lasot_sequences(root, names)
    found = {record.name for record in records}
    missing = sorted(set(names) - found)
    if missing:
        raise FileNotFoundError(f"LaSOT test split has unreadable sequences, first: {missing[:5]}")
    return records


def _annotation(info: dict, label: Path) -> tuple[np.ndarray, np.ndarray]:
    rectangles = info.get("gt_rect", info.get("bbox", info.get("rect")))
    exists = info.get("exist", info.get("visible", info.get("presence")))
    if rectangles is None or exists is None:
        raise ValueError(f"{label} must contain gt_rect/bbox and exist/visible arrays")
    return np.asarray(rectangles, np.float32).reshape(-1, 4), np.asarray(exists, dtype=bool).reshape(-1)


def anti_uav300_rgb(root: str | Path, split: str) -> list[Sequence]:
    """Anti-UAV300 RGB single-UAV tracking release.

    The official archive stores one directory per sequence, RGB image files
    directly in that directory (or in ``RGB/``), and JSON labels.  ``IR_label``
    is deliberately rejected: this release is RGB-only.
    """
    base = Path(root) / split
    if not base.exists():
        raise FileNotFoundError(f"Anti-UAV300 split directory is absent: {base}")
    records: list[Sequence] = []
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        label = next((directory / name for name in ("RGB_label.json", "label.json") if (directory / name).exists()), None)
        image_dir = next((candidate for candidate in (directory / "RGB", directory / "visible", directory) if _frames(candidate)), None)
        if label is None or image_dir is None:
            continue
        boxes, visible = _annotation(json.loads(label.read_text()), label)
        frames = _frames(image_dir)
        if len(frames) != len(boxes) or len(boxes) != len(visible):
            raise ValueError(f"Anti-UAV300 length mismatch in {directory.name}")
        records.append(Sequence(directory.name, frames, boxes, visible, {"benchmark": "anti_uav300_rgb", "label": str(label)}, ~visible))
    if not records:
        raise FileNotFoundError("expected Anti-UAV300 RGB labels (RGB_label.json or label.json) and RGB frames")
    return records


def webuav3m(root: str | Path, split: str) -> list[Sequence]:
    """WebUAV-3M V1.0 RGB Train/Val/Test layout from the official README."""
    base = Path(root) / split
    records: list[Sequence] = []
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        frames = _frames(directory / "img")
        gt, absent = directory / "groundtruth_rect.txt", directory / "absent.txt"
        if not frames or not gt.exists() or not absent.exists():
            continue
        boxes = _boxes(gt)
        absent_flags = np.loadtxt(absent, dtype=np.int32).reshape(-1)
        visible = absent_flags == 0
        if len(frames) != len(boxes) or len(boxes) != len(visible):
            raise ValueError(f"WebUAV-3M length mismatch in {directory.name}")
        records.append(
            Sequence(
                directory.name,
                frames,
                boxes,
                visible,
                {
                    "benchmark": "webuav3m_v1",
                    "scenario": (directory / "scenario.txt").read_text().strip() if (directory / "scenario.txt").exists() else "",
                },
                ~visible,
            )
        )
    if not records:
        raise FileNotFoundError("expected WebUAV-3M <split>/<video>/img, absent.txt, groundtruth_rect.txt")
    return records


def _vtuav_sequence(root: Path, name: str) -> Sequence | None:
    directory = root / name
    image_dir = next((candidate for candidate in (directory / "visible", directory / "rgb", directory / "img") if _frames(candidate)), None)
    gt = next((candidate for candidate in (directory / "groundtruth.txt", directory / "groundtruth_rect.txt", directory / "init.txt") if candidate.exists()), None)
    if image_dir is None or gt is None:
        return None
    frames, boxes = _frames(image_dir), _boxes(gt)
    visible = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    if len(frames) != len(boxes):
        raise ValueError(f"VTUAV length mismatch in {name}")
    return Sequence(name, frames, boxes, visible, {"benchmark": "vtuav_v_rgb_longterm"}, ~visible)


def vtuav_rgb_longterm(root: str | Path, split_file: str | Path | None = None) -> list[Sequence]:
    """VTUAV-V long-term RGB-only protocol used by GenerateMat_LT_RGB_only.m."""
    root = Path(root)
    if split_file is not None:
        names = _names(Path(split_file))
        sequence_root = root / "test_LT" if (root / "test_LT").exists() else root
    else:
        sequence_root = root / "test_LT" if (root / "test_LT").exists() else root
        names = [item.name for item in sorted(sequence_root.iterdir()) if item.is_dir()]
    records = [record for name in names if (record := _vtuav_sequence(sequence_root, name)) is not None]
    if not records:
        raise FileNotFoundError("expected VTUAV-V long-term visible/rgb images and groundtruth under root/test_LT or --split-file")
    return records


ADAPTERS = {
    "lasot": lasot_official_test,
    "anti_uav300_rgb": anti_uav300_rgb,
    "webuav3m": webuav3m,
    "vtuav_rgb_longterm": vtuav_rgb_longterm,
}
