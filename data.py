"""Dataset adapters for trainable RGB DDS-Mamba clips."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


@dataclass(frozen=True)
class Sequence:
    name: str; frames: tuple[Path, ...]; boxes_xywh: np.ndarray


def _image(path: Path, size: int) -> Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)) / 255.
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def _template_crop(path: Path, box: np.ndarray, size: int, context: float = 2.0) -> Tensor:
    """Target-centred square crop; image-edge coordinates are clamped safely."""
    im = Image.open(path).convert("RGB"); x, y, w, h = box.astype(float); side=max(w,h)*context; cx,cy=x+w/2,y+h/2
    crop=im.crop((max(0,cx-side/2),max(0,cy-side/2),min(im.width,cx+side/2),min(im.height,cy+side/2))).resize((size,size),Image.Resampling.BILINEAR)
    a=torch.from_numpy(np.asarray(crop,dtype=np.float32).transpose(2,0,1))/255.
    return (a-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def _box(box: np.ndarray, image_path: Path) -> Tensor:
    with Image.open(image_path) as im: w, h = im.size
    x, y, bw, bh = box.astype(np.float32); return torch.tensor([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]).clamp(0.001, .999)


def lasot_sequences(root: str | Path, split: Iterable[str] | None = None) -> list[Sequence]:
    """Read official LaSOT category/video/img layout and groundtruth.txt."""
    root, allowed = Path(root), None if split is None else set(split); found = []
    for gt in root.glob("*/*/groundtruth.txt"):
        category, video = gt.parent.parent.name, gt.parent.name; name = f"{category}-{video}"
        if allowed is not None and name not in allowed: continue
        frames = tuple(sorted((gt.parent / "img").glob("*.jpg"))); boxes = np.loadtxt(gt, delimiter=",")
        if len(frames) and len(frames) == len(boxes): found.append(Sequence(name, frames, np.asarray(boxes).reshape(-1, 4)))
    if not found: raise FileNotFoundError(f"No valid LaSOT sequences under {root}")
    return found


class TrackingClipDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, sequences: list[Sequence], clip_length: int = 8, template_size: int = 128, search_size: int = 256, seed: int = 0) -> None:
        self.sequences, self.clip_length, self.template_size, self.search_size = sequences, clip_length, template_size, search_size; self.rng = random.Random(seed)
    def __len__(self) -> int: return len(self.sequences) * 16
    def __getitem__(self, _: int) -> dict[str, Tensor]:
        seq = self.rng.choice(self.sequences); start = self.rng.randrange(0, max(1, len(seq.frames) - self.clip_length)); ids = [min(start + i, len(seq.frames) - 1) for i in range(self.clip_length)]
        template = _template_crop(seq.frames[0], seq.boxes_xywh[0], self.template_size); searches = torch.stack([_image(seq.frames[i], self.search_size) for i in ids]); boxes = torch.stack([_box(seq.boxes_xywh[i], seq.frames[i]) for i in ids])
        return {"template": template, "searches": searches, "boxes": boxes}
