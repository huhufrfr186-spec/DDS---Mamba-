"""Image/crop primitives and sequence records shared by all DDS-Mamba tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sequence:
    name: str
    rgb_frames: tuple[Path, ...]
    boxes_xywh: np.ndarray
    visible: np.ndarray
    metadata: dict
    absent: np.ndarray | None = None
    occluded: np.ndarray | None = None


def normalize(x: Tensor) -> Tensor:
    return (x - x.new_tensor(IMAGENET_MEAN)[:, None, None]) / x.new_tensor(IMAGENET_STD)[:, None, None]


def image_tensor(path: str | Path) -> Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)) / 255.


def crop_border(image: Tensor, boxes_cxcywh: Tensor, out_size: int) -> Tensor:
    """Half-open continuous crops, border replication, bilinear antialiasing.

    ``grid_sample`` has no antialias flag, so the crop is sampled at 2x and
    reduced with PyTorch's antialiased bilinear resize.  This is the exact
    preprocessing used by the locked identity encoder and is shared by every
    crop path.
    """
    b, _, h, w = image.shape; if_single = boxes_cxcywh.ndim == 1
    boxes = boxes_cxcywh[None] if if_single else boxes_cxcywh
    if len(boxes) != b: raise ValueError("one crop box per image")
    sample_size = out_size * 2
    ys = (torch.arange(sample_size, device=image.device, dtype=image.dtype) + .5) / sample_size - .5
    xs = (torch.arange(sample_size, device=image.device, dtype=image.dtype) + .5) / sample_size - .5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx,cy,bw,bh = boxes.unbind(-1); grid_x = (cx[:,None,None] + xx[None]*bw[:,None,None]) / w * 2 - 1; grid_y = (cy[:,None,None] + yy[None]*bh[:,None,None]) / h * 2 - 1
    grid = torch.stack([grid_x, grid_y], -1)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return F.interpolate(sampled, size=(out_size, out_size), mode="bilinear", align_corners=False, antialias=True)


def load_rgb_crop(path: str | Path, box_cxcywh: tuple[float,float,float,float], out_size: int) -> Tensor:
    image = image_tensor(path).unsqueeze(0); box=image.new_tensor(box_cxcywh); return normalize(crop_border(image, box, out_size)[0])


def lasot_sequences(root: str | Path, names: Iterable[str] | None = None) -> list[Sequence]:
    root, allowed = Path(root), set(names) if names is not None else None; records=[]
    for groundtruth in root.glob("*/*/groundtruth.txt"):
        category, video = groundtruth.parent.parent.name, groundtruth.parent.name; name=f"{category}-{video}"
        if allowed is not None and name not in allowed: continue
        frames=tuple(sorted((groundtruth.parent/"img").glob("*.jpg"))); boxes=np.loadtxt(groundtruth,delimiter=",").reshape(-1,4)
        if len(frames)==len(boxes): records.append(Sequence(name,frames,boxes,np.ones(len(frames),dtype=bool),{}))
    return records


@dataclass(frozen=True)
class TrainingClip:
    """One initialization plus a short unroll; images stay at native resolution."""
    name: str
    template_frame: Path
    template_box_xywh: np.ndarray
    frames: tuple[Path, ...]
    boxes_xywh: np.ndarray
    visible: np.ndarray
    absent: np.ndarray
    occluded: np.ndarray


class TrainingClips(torch.utils.data.Dataset):
    """Deterministic native-image clips for the controller-driven trainer.

    The dataset never supplies a ground-truth-centred search crop.  Crop choice
    is made by the online state in ``train_v1.py`` exactly as at test time.
    """
    def __init__(self, sequences: list[Sequence], length: int = 16, seed: int = 0, clips_per_sequence: int = 16) -> None:
        self.sequences, self.length, self.seed, self.clips_per_sequence = sequences, length, seed, clips_per_sequence
        self.epoch = 0
        if not sequences:
            raise ValueError("training split did not resolve to any readable sequence")

    def set_epoch(self, epoch: int) -> None:
        """Deterministically resample short unroll start positions each epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.sequences) * self.clips_per_sequence

    def __getitem__(self, index: int) -> TrainingClip:
        sequence = self.sequences[index % len(self.sequences)]
        rng = random.Random(self.seed + 1_000_003 * index + 104_729 * self.epoch)
        valid_starts = [i for i, shown in enumerate(sequence.visible) if shown and i + 1 < len(sequence.rgb_frames)]
        if not valid_starts:
            raise ValueError(f"{sequence.name} has no visible initialization frame")
        start = valid_starts[rng.randrange(len(valid_starts))]
        end = min(len(sequence.rgb_frames), start + 1 + self.length)
        absent = np.asarray(sequence.absent if sequence.absent is not None else ~sequence.visible, dtype=bool)
        occluded = np.asarray(sequence.occluded if sequence.occluded is not None else np.zeros(len(sequence.visible), dtype=bool), dtype=bool)
        return TrainingClip(
            sequence.name,
            sequence.rgb_frames[start],
            np.asarray(sequence.boxes_xywh[start], dtype=np.float32),
            tuple(sequence.rgb_frames[start + 1 : end]),
            np.asarray(sequence.boxes_xywh[start + 1 : end], dtype=np.float32),
            np.asarray(sequence.visible[start + 1 : end], dtype=bool),
            absent[start + 1 : end],
            occluded[start + 1 : end],
        )
